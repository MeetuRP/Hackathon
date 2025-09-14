import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL,
    NEXT_PUBLIC_ANALYTICS_ID: process.env.NEXT_PUBLIC_ANALYTICS_ID,
    NEXT_PUBLIC_BACKEND_URL: "https://hackathon-i2pv.onrender.com/",
  },
  reactStrictMode: true,
  swcMinify: true,
  /* config options here */
};

export default nextConfig;
