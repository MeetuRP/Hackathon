"use client";

import { useEffect, useState } from "react";
import { toast } from "sonner";
import { TripTable, type Trip } from "./trip-table";

export default function Home() {
  const [data, setData] = useState<Trip[]>([]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await fetch("/api/schedule/static");
        const json = await res.json();
        console.log("raw response:", json);

        if (!json) {
          console.error("No JSON returned from API");
          return;
        }

        if (json.success) {
          toast.success(json.message);

          const schedule = json.schedule;
          console.log("schedule (raw):", schedule);

          // Normalize: if schedule is already an array, use it.
          // If it's an object keyed by route/trip ids (e.g. { "1": [...], "2": [...] }),
          // convert to an array of trips via Object.values(...).flat()
          const normalized: Trip[] = Array.isArray(schedule)
            ? schedule
            : Object.values(schedule).flat();

          console.log("schedule (normalized):", normalized);
          setData(normalized);
          return;
        }

        toast.error(json.message ?? "Failed to load schedule");
      } catch (error) {
        console.error("Error fetching data:", error);
        toast.error("Failed to fetch schedule");
      }
    };

    fetchData();
  }, []);

  return (
    <div className="w-[70dvw] h-full mt-[-60vh] flex flex-col items-center justify-center gap-3">
      <TripTable data={data} />
    </div>
  );
}
