"use client";
import { useEffect, useState } from "react";
import Image from "next/image";
import { RouteTable } from "./data-table";
import { toast } from "sonner";
import { TypewriterEffect } from "@/components/ui/typewriter-effect";
import { Header } from "@/components/ui/header";

export default function Home() {
  // get

  const [data, setData] = useState([]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch("/api/getroutes");
        const data = await response.json();
        if (data.success) {
          toast.success(data.message);
          const routes = data.routes;
          setData(routes);
          return;
        }
        toast.error(data.message);
      } catch (error) {
        console.error("Error fetching data:", error);
      }
    };
    fetchData();
  }, []);

  return (
    <div className="w-[70dvw] h-full flex flex-col items-center justify-center gap-3">
      <Header />
      <RouteTable data={data} />
    </div>
  );
}
