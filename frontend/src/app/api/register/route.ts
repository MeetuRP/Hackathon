import { NextResponse, NextRequest } from "next/server";
import { success } from "zod";

export async function POST(request: NextRequest) {
  const body = await request.json();
  const { username, email, password } = body;
  // Perform register logic here
  // call to backend API for authentication
  console.log(`${process.env.NEXT_PUBLIC_BACKEND_URL}register`);
  const response = await fetch(
    `${process.env.NEXT_PUBLIC_BACKEND_URL}register`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ username, email, password }),
    }
  );
  const data = await response.json();
  if (!response.ok) {
    return NextResponse.json(
      { message: data.message || "register failed" },
      { status: response.status }
    );
  }

  return NextResponse.json({ success: data.success, message: data.message });
}

export async function GET(request: NextRequest) {
  return NextResponse.json(
    { success: false, message: "GET method not supported for register" },
    { status: 405 }
  );
}
