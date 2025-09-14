"use client";

import { ColumnDef } from "@tanstack/react-table";

export type Stop = {
  stop_id: number;
  stop_name: string;
  latitude: number;
  longitude: number;
};

export type Route = {
  route_id: number;
  route_name: string;
  stops: Stop[];
};

export const stopColumns: ColumnDef<Stop>[] = [
  {
    accessorKey: "stop_id",
    header: "Stop ID",
  },
  {
    accessorKey: "stop_name",
    header: "Stop Name",
  },
  {
    accessorKey: "latitude",
    header: "Latitude",
  },
  {
    accessorKey: "longitude",
    header: "Longitude",
  },
];

export const routeColumns: ColumnDef<Route>[] = [
  {
    accessorKey: "route_id",
    header: "Route ID",
  },
  {
    accessorKey: "route_name",
    header: "Route Name",
  },
  {
    accessorKey: "stops",
    header: "Stops Count",
    cell: ({ row }) => row.original.stops.length, // shows number of stops
  },
];
