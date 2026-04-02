/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        primary: '#03dac6',
        secondary: '#bb86fc',
        error: '#cf6679',
        success: '#4caf50',
        warning: '#ff9800',
      },
    },
  },
  plugins: [],
}
