/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'liberal': '#3b82f6',
        'fascist': '#ef4444',
        'hitler': '#dc2626',
      },
      fontFamily: {
        'game': ['Cinzel', 'serif'],
      },
    },
  },
  plugins: [],
}