// Face Recognition Web App - Vietnamese

const API_BASE = '';

// Tab Navigation
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById(btn.dataset.tab).classList.add('active');

        if (btn.dataset.tab === 'people') {
            loadPeople();
        }
    });
});

// ============== Recogn?e ==============

let recognizeImage = null;
const recognizeDrop = document.getElementById('recognize-drop');
const recognizeFile = document.getElementById('recognize-file');
const recognizePreview = document.getElementById('recognize-preview');
const recognizeImg = document.getElementById('recognize-img');
const recognizeCanvas = document.getElementById('recognize-canvas');
const recognizeBtn = document.getElementById('recognize-btn');
const thresholdSlider = document.getElementById('recognize-threshold');
const thresholdValue = document.getElementById('threshold-value');
const recognizeResults = document.getElementById('recognize-results');

thresholdSlider.addEventListener('input', (e) => {
    thresholdValue.textContent = e.target.value;
});

recognizeDrop.addEventListener('click', () => recognizeFile.click());
recognizeDrop.addEventListener('dragover', (e) => {
    e.preventDefault();
    recognizeDrop.style.borderColor = '#03dac6';
});
recognizeDrop.addEventListener('dragleave', () => {
    recognizeDrop.style.borderColor = '#555';
});
recognizeDrop.addEventListener('drop', async (e) => {
    e.preventDefault();
    recognizeDrop.style.borderColor = '#555';
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        await loadRecognizeImage(file);
    }
});

recognizeFile.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (file) {
        await loadRecognizeImage(file);
    }
});

async function loadRecognizeImage(file) {
    const img = new Image();
    img.onload = () => {
        recognizeImage = img;
        recognizeImg.src = img.src;
        recognizePreview.style.display = 'block';
        recognizeBtn.disabled = false;
        recognizeResults.innerHTML = '';
        // Resize canvas
        recognizeCanvas.width = img.naturalWidth;
        recognizeCanvas.height = img.naturalHeight;
    };
    img.onerror = () => {
        showAlert(recognizeResults, 'Không thể tải ảnh', 'error');
    };
    img.src = URL.createObjectURL(file);
}

recognizeBtn.addEventListener('click', async () => {
    if (!recognizeImage) return;

    recognizeBtn.disabled = true;
    recognizeBtn.innerHTML = '<span class="loading"></span> Đang xử lý...';
    recognizeResults.innerHTML = '';

    try {
        const blob = await fetch(recognizeImg.src).then(r => r.blob());

        const formData = new FormData();
        formData.append('image', blob, 'image.jpg');
        formData.append('threshold', thresholdSlider.value);

        const response = await fetch(`${API_BASE}/api/recognize`, {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        recognizeBtn.disabled = false;
        recognizeBtn.textContent = '🔍 Nhận Diện';

        if (result.success) {
            drawRecognizeResults(result.data.faces);
            showAlert(recognizeResults, `Tìm thấy ${result.data.faces.length} khuôn mặt`, 'success');
        } else {
            showAlert(recognizeResults, result.error || 'Nhận diện thất bại', 'error');
        }
    } catch (error) {
        recognizeBtn.disabled = false;
        recognizeBtn.textContent = '🔍 Nhận Diện';
        showAlert(recognizeResults, 'Lỗi: ' + error.message, 'error');
    }
});

function drawRecognizeResults(faces) {
    const ctx = recognizeCanvas.getContext('2d');
    ctx.clearRect(0, 0, recognizeCanvas.width, recognizeCanvas.height);
    ctx.drawImage(recognizeImg, 0, 0);

    faces.forEach(face => {
        const [x1, y1, x2, y2] = face.bbox;
        const width = x2 - x1;
        const height = y2 - y1;

        // Draw box
        ctx.strokeStyle = face.matched ? '#03dac6' : '#cf6679';
        ctx.lineWidth = 3;
        ctx.strokeRect(x1, y1, width, height);

        // Draw name and confidence
        const label = face.name === 'unknown' ? 'Không xác định' : `${face.name} (${(face.confidence * 100).toFixed(1)}%)`;
        ctx.font = 'bold 18px Arial';
        ctx.fillStyle = face.matched ? '#03dac6' : '#cf6679';
        ctx.fillText(label, x1, y1 > 25 ? y1 - 10 : y1 + 25);

        // Draw quality
        if (face.quality_score) {
            ctx.font = '12px Arial';
            ctx.fillStyle = '#888';
            ctx.fillText(`Chất lượng: ${(face.quality_score * 100).toFixed(0)}%`, x1, y2 + 15);
        }
    });
}

// ============== Realtime ==============

let realtimeStream = null;
let realtimeRunning = false;
let realtimeInterval = null;

const startRealtimeBtn = document.getElementById('start-realtime-btn');
const stopRealtimeBtn = document.getElementById('stop-realtime-btn');
const realtimeVideo = document.getElementById('realtime-video');
const realtimeCanvas = document.getElementById('realtime-canvas');
const realtimeResults = document.getElementById('realtime-results');
const realtimeThresholdSlider = document.getElementById('realtime-threshold');
const realtimeThresholdValue = document.getElementById('realtime-threshold-value');

realtimeThresholdSlider.addEventListener('input', (e) => {
    realtimeThresholdValue.textContent = e.target.value;
});

startRealtimeBtn.addEventListener('click', startRealtime);
stopRealtimeBtn.addEventListener('click', stopRealtime);

async function startRealtime() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: 'user' }
        });
        realtimeStream = stream;
        realtimeVideo.srcObject = stream;

        realtimeVideo.onloadedmetadata = () => {
            realtimeCanvas.width = realtimeVideo.videoWidth;
            realtimeCanvas.height = realtimeVideo.videoHeight;
        };

        startRealtimeBtn.disabled = true;
        stopRealtimeBtn.disabled = false;
        realtimeRunning = true;

        // Start detection loop
        realtimeInterval = setInterval(detectRealtimeFrame, 200); // 5 FPS detection

        showAlert(realtimeResults, 'Realtime đã bắt đầu', 'info');
    } catch (error) {
        showAlert(realtimeResults, 'Không thể truy cập webcam: ' + error.message, 'error');
    }
}

function stopRealtime() {
    realtimeRunning = false;

    if (realtimeInterval) {
        clearInterval(realtimeInterval);
        realtimeInterval = null;
    }

    if (realtimeStream) {
        realtimeStream.getTracks().forEach(track => track.stop());
        realtimeStream = null;
        realtimeVideo.srcObject = null;
    }

    startRealtimeBtn.disabled = false;
    stopRealtimeBtn.disabled = true;

    const ctx = realtimeCanvas.getContext('2d');
    ctx.clearRect(0, 0, realtimeCanvas.width, realtimeCanvas.height);

    showAlert(realtimeResults, 'Realtime đã dừng', 'info');
}

async function detectRealtimeFrame() {
    if (!realtimeRunning || !realtimeStream) return;

    // Capture frame
    const ctx = realtimeCanvas.getContext('2d');
    ctx.drawImage(realtimeVideo, 0, 0);

    try {
        const blob = await new Promise(resolve => realtimeCanvas.toBlob(resolve, 'image/jpeg', 0.8));

        const formData = new FormData();
        formData.append('image', blob, 'frame.jpg');
        formData.append('threshold', realtimeThresholdSlider.value);

        const response = await fetch(`${API_BASE}/api/recognize`, {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.success && result.data.faces.length > 0) {
            drawRealtimeFaces(result.data.faces);
        }
    } catch (error) {
        console.error('Realtime detection error:', error);
    }
}

function drawRealtimeFaces(faces) {
    const ctx = realtimeCanvas.getContext('2d');
    ctx.clearRect(0, 0, realtimeCanvas.width, realtimeCanvas.height);
    ctx.drawImage(realtimeVideo, 0, 0);

    faces.forEach(face => {
        const [x1, y1, x2, y2] = face.bbox;
        const width = x2 - x1;
        const height = y2 - y1;

        ctx.strokeStyle = face.matched ? '#03dac6' : '#cf6679';
        ctx.lineWidth = 3;
        ctx.strokeRect(x1, y1, width, height);

        const label = face.name === 'unknown' ? '??? ' : `${face.name} `;
        ctx.font = 'bold 16px Arial';
        ctx.fillStyle = face.matched ? '#03dac6' : '#cf6679';
        ctx.fillText(label + (face.confidence * 100).toFixed(0) + '%', x1, y1 > 20 ? y1 - 5 : y1 + 20);
    });
}

// ============== Register ==============

const registerName = document.getElementById('register-name');
const registerFiles = document.getElementById('register-files');
const registerDrop = document.getElementById('register-drop');
const registerPreviews = document.getElementById('register-previews');
const registerBtn = document.getElementById('register-btn');
const clearRegisterBtn = document.getElementById('clear-register-btn');
const registerResults = document.getElementById('register-results');

let registerImages = [];

registerDrop.addEventListener('click', () => registerFiles.click());
registerDrop.addEventListener('dragover', (e) => {
    e.preventDefault();
    registerDrop.style.borderColor = '#03dac6';
});
registerDrop.addEventListener('dragleave', () => {
    registerDrop.style.borderColor = '#555';
});
registerDrop.addEventListener('drop', async (e) => {
    e.preventDefault();
    registerDrop.style.borderColor = '#555';
    const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/'));
    await addRegisterImages(files);
});

registerFiles.addEventListener('change', async (e) => {
    const files = Array.from(e.target.files);
    await addRegisterImages(files);
});

async function addRegisterImages(files) {
    for (const file of files) {
        try {
            const img = await loadImage(file);
            registerImages.push({ file, img });
        } catch (error) {
            console.error('Không thể tải ảnh:', error);
        }
    }
    updateRegisterPreviews();
    validateRegisterForm();
}

function loadImage(file) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve({ file, img, url: URL.createObjectURL(file) });
        img.onerror = reject;
        img.src = URL.createObjectURL(file);
    });
}

function updateRegisterPreviews() {
    registerPreviews.innerHTML = '';
    registerImages.forEach((item, index) => {
        const div = document.createElement('div');
        div.className = 'preview-item';
        div.innerHTML = `
            <img src="${item.url}" alt="Ảnh ${index + 1}" style="height: 120px; object-fit: cover;">
            <button class="remove-img" data-index="${index}" title="Xóa ảnh">✕</button>
        `;
        registerPreviews.appendChild(div);
    });

    // Add remove handlers
    document.querySelectorAll('.remove-img').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const idx = parseInt(e.target.dataset.index);
            registerImages.splice(idx, 1);
            updateRegisterPreviews();
            validateRegisterForm();
        });
    });
}

function validateRegisterForm() {
    const name = registerName.value.trim();
    const hasImages = registerImages.length > 0;
    registerBtn.disabled = !(name && hasImages);
}

registerName.addEventListener('input', validateRegisterForm);

clearRegisterBtn.addEventListener('click', () => {
    registerImages = [];
    registerName.value = '';
    registerPreviews.innerHTML = '';
    registerResults.innerHTML = '';
    validateRegisterForm();
});

registerBtn.addEventListener('click', async () => {
    const name = registerName.value.trim();
    if (!name || registerImages.length === 0) return;

    registerBtn.disabled = true;
    registerBtn.innerHTML = '<span class="loading"></span> Đang đăng ký...';
    registerResults.innerHTML = '';

    try {
        const formData = new FormData();
        formData.append('name', name);
        registerImages.forEach(item => {
            formData.append('images', item.file);
        });

        const response = await fetch(`${API_BASE}/api/register`, {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        registerBtn.disabled = false;
        registerBtn.textContent = '✅ Đăng Ký';

        if (result.success) {
            showAlert(registerResults, `Đã đăng ký ${name} với ${result.data.embeddings_extracted} ảnh khuôn mặt!`, 'success');
            clearRegisterBtn.click();
            loadPeople();
        } else {
            showAlert(registerResults, 'Lỗi: ' + (result.error || 'Đăng ký thất bại'), 'error');
            registerBtn.textContent = '✅ Đăng Ký';
        }
    } catch (error) {
        registerBtn.disabled = false;
        registerBtn.textContent = '✅ Đăng Ký';
        showAlert(registerResults, 'Lỗi kết nối: ' + error.message, 'error');
    }
});

// ============== People ==============

const loadPeopleBtn = document.getElementById('load-people-btn');
const deleteSelectedBtn = document.getElementById('delete-selected-btn');
const peopleList = document.getElementById('people-list');

async function loadPeople() {
    loadPeopleBtn.disabled = true;
    loadPeopleBtn.innerHTML = '<span class="loading"></span> Đang tải...';
    peopleList.innerHTML = '';

    try {
        const response = await fetch(`${API_BASE}/api/people`);
        const result = await response.json();

        if (result.success) {
            displayPeople(result.data.people);
        } else {
            peopleList.innerHTML = '<p class="info-text">Không thể tải danh sách</p>';
        }
    } catch (error) {
        peopleList.innerHTML = `<p class="info-text">Lỗi: ${error.message}</p>`;
    } finally {
        loadPeopleBtn.disabled = false;
        loadPeopleBtn.textContent = '🔄 Tải Danh Sách';
    }
}

function displayPeople(people) {
    peopleList.innerHTML = '';

    if (people.length === 0) {
        peopleList.innerHTML = '<p class="info-text">Chưa có ai được đăng ký.</p>';
        return;
    }

    people.forEach(person => {
        const card = document.createElement('div');
        card.className = 'person-card';
        card.innerHTML = `
            <input type="checkbox" class="person-select" data-name="${person}">
            <div class="person-name">${person}</div>
            <div class="person-stats">ID: ${person.substring(0, 8)}...</div>
        `;
        peopleList.appendChild(card);
    });

    // Show delete button if any selected
    document.querySelectorAll('.person-select').forEach(cb => {
        cb.addEventListener('change', () => {
            const anySelected = document.querySelectorAll('.person-select:checked').length > 0;
            deleteSelectedBtn.style.display = anySelected ? 'block' : 'none';
        });
    });

    deleteSelectedBtn.onclick = async () => {
        if (!confirm('Xóa các người đã chọn? Hành động này không thể hoàn tác.')) return;

        const selected = Array.from(document.querySelectorAll('.person-select:checked'))
            .map(cb => cb.dataset.name);

        for (const name of selected) {
            try {
                await fetch(`${API_BASE}/api/person/${encodeURIComponent(name)}`, {
                    method: 'DELETE'
                });
            } catch (error) {
                console.error('Không thể xóa:', name, error);
            }
        }

        loadPeople();
        deleteSelectedBtn.style.display = 'none';
    };
}

// ============== Utilities ==============

function showAlert(container, message, type = 'info') {
    const colors = {
        success: '#d4edda',
        error: '#f8d7da',
        info: '#d1ecf1'
    };
    const textColors = {
        success: '#155724',
        error: '#721c24',
        info: '#0c5460'
    };

    const alert = document.createElement('div');
    alert.className = `alert alert-${type}`;
    alert.innerHTML = message;
    container.innerHTML = '';
    container.appendChild(alert);

    if (type !== 'error') {
        setTimeout(() => {
            alert.style.transition = 'opacity 0.5s';
            alert.style.opacity = '0';
            setTimeout(() => alert.remove(), 500);
        }, 5000);
    }
}

// Check API connection on load
document.addEventListener('DOMContentLoaded', async () => {
    try {
        const response = await fetch(`${API_BASE}/api/health`);
        const data = await response.json();
        if (!data.success) {
            showAlert(document.body, 'Không thể kết nối đến API. Hãy đảm bảo server đang chạy.', 'error');
        }
    } catch (error) {
        // Silent fail, will show on first API call
    }
});
