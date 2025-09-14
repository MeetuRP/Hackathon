import { NextRequest, NextResponse } from "next/server";

export async function GET(request: NextRequest) {
  // Fetch schedule from backend API
  const response = await fetch(
    `${process.env.NEXT_PUBLIC_BACKEND_URL}api/schedule/static`,
    {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    }
  );
  const data = await response.json();
  if (!response.ok || !data) {
    return NextResponse.json(
      { message: data.message || "Failed to fetch schedule" },
      { status: response.status }
    );
  }
  if (data === 0) {
    return NextResponse.json({
      success: false,
      message: "No schedule found",
      schedule: [],
    });
  }
  return NextResponse.json({
    success: true,
    message: "Schedule found",
    schedule: data,
  });
}
