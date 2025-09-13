import { NextResponse , NextRequest } from "next/server";

export async function POST(request: NextRequest) {
  const body = await request.json();
  const { email, password } = body; 
  // Perform login logic here
  // call to backend API for authentication
  const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/api/login`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ email, password }),
  });
    const data = await response.json();
    if (!response.ok) {
      return NextResponse.json({ message: data.message || "Login failed" }, { status: response.status });
    }
    // On successful login, we need to store user information
    data.token && (NextResponse.next().cookies.set('token', data.token, { httpOnly: true, secure: true, path: '/' }));
  return NextResponse.json({ message: "Login successful" });
}



export async function GET(request: NextRequest) {
  return NextResponse.json({ message: "GET method not supported for login" }, { status: 405 });
}