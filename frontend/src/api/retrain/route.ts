import { NextRequest , NextResponse } from "next/server";



export async function POST(request: NextRequest) {
  const data = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/api/retrain_model`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
  });
  const res = await data.json();
}