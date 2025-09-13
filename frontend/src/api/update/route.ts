import { time } from "console";
import { NextRequest , NextResponse } from "next/server";



export async function POST(request: NextRequest) {
  const body = await request.json();
  const { id , route_id , trip_id, bus_id, latitude , longitude , speed_kmh , occupacy  } = body;
  // Send data to backend API
  const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/api/live_update`, { 
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      route_id,
      trip_id,
      bus_id,
      latitude,
      longitude,
      speed_kmh,
      occupacy,
      timestamp: new Date().toISOString()
  }),
  });
  const data = await response.json();
  if (!response.ok || !data) {
    return NextResponse.json({ message: data.message || "Failed to send update" }, { status: response.status });
  } 
  return NextResponse.json({ success: true , message: "Update sent successfully" , update: data });
}