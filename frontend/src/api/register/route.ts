import { NextResponse , NextRequest } from "next/server";

export async function POST(request: NextRequest) {
  const body = await request.json();
  const { email, password } = body;
  // Perform registration logic here
  // call to backend API for user registration
  const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/api/register`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ email, password }),
  });
  const data = await response.json();
  if (!response.ok) {
    return NextResponse.json({ message: data.message || "Registration failed" }, { status: response.status });
  }
  // On successful registration, we need to store user information
  return NextResponse.json({ message: "Registration successful" });
}
