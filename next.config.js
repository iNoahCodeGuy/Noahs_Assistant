/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  env: {
    PYTHON_BACKEND_URL: process.env.PYTHON_BACKEND_URL || 'http://localhost:8000',
  },
}

module.exports = nextConfig
