import {NextRequest, NextResponse} from 'next/server';


export async function GET(request: NextRequest) {
  // get a id from body
  const body = await request.json();
  const { id } = body;
  // Fetch routes from backend API
  const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/api/predictions/${id}`, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
    },
  });
  const data = await response.json();
  if (!response.ok || !data) {
    return NextResponse.json({ message: data.message || "Failed to fetch route predictions" }, { status: response.status });
  }
  if(data === 0){
    return NextResponse.json({ success: false , message: "No route predictions found" , route: [] });
  }
  return NextResponse.json({ success: true , message: "Route predictions found" , routes: data });
}