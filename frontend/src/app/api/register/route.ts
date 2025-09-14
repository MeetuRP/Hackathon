import { NextResponse, NextRequest } from "next/server";
import { success } from "zod";

export async function POST(request: NextRequest) {
  const body = await request.json();
  const { username, email, password } = body;
  // Perform register logic here
  // call to backend API for authentication
  const response = await fetch(
    `${process.env.NEXT_PUBLIC_BACKEND_URL}/api/register`,
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
  // On successful register, we need to store user information
  data.token &&
    NextResponse.next().cookies.set("token", data.token, {
      httpOnly: true,
      secure: true,
      path: "/",
    });
  return NextResponse.json({ success: true, message: "register successful" });
}

export async function GET(request: NextRequest) {
  return NextResponse.json(
    { success: false, message: "GET method not supported for register" },
    { status: 405 }
  );
}
