"use client";
import { useParams } from "next/navigation";
import React, { useState, useEffect } from "react";
import { PredictionTable, type RoutePrediction } from "./prediction-table";
import { toast } from "sonner";

const Page = () => {
  const { id } = useParams();
  const [data, setData] = useState<RoutePrediction | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!id) return;

    const fetchData = async () => {
      try {
        const res = await fetch(`/api/predictions/`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ id }),
        });
        const json = await res.json();
        console.log("response:", json);

        if (json.success) {
          toast.success(json.message);
          setData(json.routes); // single object
        } else {
          toast.error(json.message ?? "Failed to load prediction");
        }
      } catch (err) {
        toast.error("Failed to fetch prediction");
        console.error("Error fetching prediction:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [id]);

  if (loading) return <div>Loading...</div>;
  if (!data) return <div>No prediction found for {id}</div>;

  return (
    <div className="w-[70dvw] h-full flex flex-col items-center justify-center">
      <PredictionTable data={[data]} />
    </div>
  );
};

export default Page;
