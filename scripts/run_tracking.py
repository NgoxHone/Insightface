#!/usr/bin/env python3
"""
Real-time Face Recognition Pipeline with Tracking

Run the YOLO + tracking pipeline on a video source:
- Webcam (default)
- RTSP stream
- Video file

Usage:
    python scripts/run_tracking.py
    python scripts/run_tracking.py --source 0
    python scripts/run_tracking.py --source rtsp://...
    python scripts/run_tracking.py --source video.mp4 --output output.mp4
"""

import argparse
import logging
import time
from pathlib import Path
import sys
import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from tracking.pipeline import FaceTrackingPipeline
from tracking.video_source import VideoSource, create_video_source
from recognition.database import FaceDatabase

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.LOG_FILE) if config.LOG_FILE.parent.exists() else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Realtime face tracking pipeline")
    parser.add_argument(
        '--source',
        type=str,
        default='0',
        help='Video source: camera index (0), RTSP URL, or video file path'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output video file path (optional)'
    )
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Run without displaying window (headless)'
    )
    parser.add_argument(
        '--resolution',
        type=str,
        default='640x480',
        help='Frame resolution (WxH), e.g., 640x480, 1280x720'
    )
    parser.add_argument(
        '--database',
        type=str,
        default=None,
        help='Path to face database (default: config.DATABASE_PATH)'
    )
    parser.add_argument(
        '--frame-skip',
        type=int,
        default=config.FRAME_SKIP,
        help='Process every N frames (1=no skip)'
    )
    parser.add_argument(
        '--recognition-interval',
        type=float,
        default=config.RECOGNITION_INTERVAL,
        help='Seconds between re-recognitions for same track'
    )
    parser.add_argument(
        '--yolo-model',
        type=str,
        default=config.YOLO_MODEL_NAME,
        help='YOLO model name/path (yolov8n.pt, yolov8s.pt, etc.)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=config.YOLO_DEVICE,
        help='Inference device: cpu, cuda, cuda:0, etc.'
    )
    parser.add_argument(
        '--tracking',
        action='store_true',
        default=config.TRACKING_ENABLED,
        help='Enable tracking (default: enabled)'
    )
    parser.add_argument(
        '--no-tracking',
        action='store_false',
        dest='tracking',
        help='Disable tracking (detection only)'
    )
    parser.add_argument(
        '--max-duration',
        type=int,
        default=None,
        help='Maximum run duration in seconds (optional)'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Parse source
    try:
        source = int(args.source) if args.source.isdigit() else args.source
    except:
        source = args.source

    logger.info("=" * 60)
    logger.info("Starting Realtime Face Tracking Pipeline")
    logger.info(f"Source: {args.source}")
    logger.info(f"Resolution: {args.resolution}")
    logger.info(f"YOLO model: {args.yolo_model}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Tracking: {args.tracking}")
    logger.info(f"Frame skip: {args.frame_skip}")
    logger.info(f"Recognition interval: {args.recognition_interval}s")
    logger.info("=" * 60)

    # Initialize database
    db_path = Path(args.database) if args.database else config.DATABASE_PATH
    database = FaceDatabase(db_path)
    logger.info(f"Loaded database with {len(database.get_all_people())} people")

    # Initialize pipeline
    from tracking.pipeline import PipelineConfig
    pipeline_config = PipelineConfig(
        frame_skip=args.frame_skip,
        recognition_interval=args.recognition_interval,
        enable_tracking=args.tracking
    )

    pipeline = FaceTrackingPipeline(database=database, config_override=pipeline_config)

    # Override YOLO model if needed (requires recreating detector)
    if args.yolo_model != config.YOLO_MODEL_NAME:
        from tracking.detector import YOLODetector
        pipeline.detector = YOLODetector(
            model_name=args.yolo_model,
            device=args.device
        )
        if args.tracking:
            from tracking.tracker import ByteTrackWrapper
            pipeline.tracker = ByteTrackWrapper(
                model_name=args.yolo_model,
                device=args.device
            )
        logger.info(f"Overrode YOLO model to: {args.yolo_model}")

    # Open video source
    video = create_video_source(
        source=source,
        resolution=args.resolution
    )

    # Setup output video writer if needed
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(out_path),
            fourcc,
            video.actual_fps if video.actual_fps > 0 else 30,
            (video.actual_width, video.actual_height)
        )
        logger.info(f"Writing output to: {args.output}")

    # Main loop
    start_time = time.time()
    frame_count = 0
    try:
        while True:
            ret, frame = video.read()
            if not ret:
                logger.info("End of video stream")
                break

            frame_count += 1

            # Process frame
            output = pipeline.process_frame(frame)

            # Annotate
            annotated = pipeline.annotate_frame(frame, output)

            # Write output
            if writer:
                writer.write(annotated)

            # Display
            if not args.no_display:
                cv2.imshow('Face Tracking', annotated)

                # Press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("User interrupted")
                    break
            else:
                # Headless: log stats periodically
                if frame_count % 100 == 0:
                    stats = pipeline.get_stats()
                    logger.info(f"Processed {frame_count} frames, fps={stats['fps']:.1f}")

            # Check max duration
            if args.max_duration and (time.time() - start_time) > args.max_duration:
                logger.info(f"Reached max duration ({args.max_duration}s)")
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error in main loop: {e}", exc_info=True)
    finally:
        # Cleanup
        video.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        # Final stats
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        stats = pipeline.get_stats()

        logger.info("=" * 60)
        logger.info("Pipeline finished")
        logger.info(f"Total frames: {frame_count}")
        logger.info(f"Total time: {elapsed:.2f}s")
        logger.info(f"Average FPS: {avg_fps:.2f}")
        logger.info(f"Final stats: {stats}")
        logger.info("=" * 60)


if __name__ == '__main__':
    main()
