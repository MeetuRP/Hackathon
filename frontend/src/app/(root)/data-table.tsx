"use client";

import React, { useState } from "react";
import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  useReactTable,
} from "@tanstack/react-table";
import Link from "next/link";

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

// Columns for stops
export const stopColumns: ColumnDef<Stop>[] = [
  { accessorKey: "stop_id", header: "Stop ID" },
  { accessorKey: "stop_name", header: "Stop Name" },
  { accessorKey: "latitude", header: "Latitude" },
  { accessorKey: "longitude", header: "Longitude" },
];

// Columns for routes (expandable)
export const routeColumns: ColumnDef<Route>[] = [
  {
    id: "expander",
    header: () => "", // blank header but keeps table aligned
    cell: ({ row }) =>
      row.getCanExpand() ? (
        <button
          onClick={row.getToggleExpandedHandler()}
          className="px-2 py-1 border rounded bg-zinc-800 hover:bg-zinc-900 transition"
        >
          {row.getIsExpanded() ? "−" : "+"}
        </button>
      ) : null,
  },
  { accessorKey: "route_id", header: "Route ID" },
  { accessorKey: "route_name", header: "Route Name" },
  {
    accessorKey: "stops",
    header: "Stops Count",
    cell: ({ row }) => row.original.stops.length,
  },
];

// RouteTable component
export function RouteTable({ data }: { data: Route[] }) {
  const table = useReactTable({
    data,
    columns: routeColumns,
    getCoreRowModel: getCoreRowModel(),
    getRowCanExpand: () => true,
  });

  return (
    <table className="min-w-full border border-gray-300">
      <thead className="bg-zinc-900">
        {table.getHeaderGroups().map((headerGroup) => (
          <tr key={headerGroup.id}>
            {headerGroup.headers.map((header) => (
              <th key={header.id} className="border px-2 py-1 text-left">
                {flexRender(
                  header.column.columnDef.header,
                  header.getContext()
                )}
              </th>
            ))}
          </tr>
        ))}
      </thead>
      <tbody>
        {table.getRowModel().rows.map((row) => (
          <React.Fragment key={row.id}>
            <tr className="hover:bg-zinc-900 transition">
              {row.getVisibleCells().map((cell) => (
                <td key={cell.id} className="border px-2 py-1">
                  <Link href={`/prediction/${row.original.route_id}`}>
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </Link>
                </td>
              ))}
            </tr>

            {/* Expanded row with animation */}
            <tr>
              <td
                colSpan={row.getVisibleCells().length}
                className="p-0 border-none"
              >
                <div
                  className={`overflow-hidden transition-all duration-300 ease-in-out ${
                    row.getIsExpanded()
                      ? "max-h-[500px] opacity-100"
                      : "max-h-0 opacity-0"
                  }`}
                >
                  <div className="ml-6 mr-6 mt-2 mb-2 border rounded bg-zinc-800 shadow-sm">
                    <table className="min-w-[90%] border border-gray-300 mx-auto">
                      <thead className="bg-zinc-800">
                        <tr>
                          {stopColumns.map((col) => (
                            <th
                              key={
                                col.id?.toString() ||
                                col.accessorKey?.toString()
                              }
                              className="border px-2 py-1 text-left"
                            >
                              {typeof col.header === "string"
                                ? col.header
                                : (col.header as any)()}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {row.original.stops.map((stop) => (
                          <tr key={stop.stop_id} className=" transition">
                            <td className="border px-2 py-1">{stop.stop_id}</td>
                            <td className="border px-2 py-1">
                              {stop.stop_name}
                            </td>
                            <td className="border px-2 py-1">
                              {stop.latitude}
                            </td>
                            <td className="border px-2 py-1">
                              {stop.longitude}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </td>
            </tr>
          </React.Fragment>
        ))}
      </tbody>
    </table>
  );
}
