"use client";

import React from "react";
import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  useReactTable,
} from "@tanstack/react-table";

// ---------- Types ----------
export type Arrival = {
  stop_id: number;
  arrival_time: string;
};

export type Trip = {
  trip_id: string;
  bus_id: string;
  depart_time: string;
  arrivals: Arrival[];
};

// ---------- Columns ----------
export const arrivalColumns: ColumnDef<Arrival>[] = [
  { accessorKey: "stop_id", header: "Stop ID" },
  { accessorKey: "arrival_time", header: "Arrival Time" },
];

export const tripColumns: ColumnDef<Trip>[] = [
  {
    id: "expander",
    header: () => "",
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
  { accessorKey: "trip_id", header: "Trip ID" },
  { accessorKey: "bus_id", header: "Bus ID" },
  { accessorKey: "depart_time", header: "Departure Time" },
  {
    accessorKey: "arrivals",
    header: "Stops Count",
    cell: ({ row }) => row.original.arrivals.length,
  },
];

// ---------- TripTable ----------
export function TripTable({ data = [] }: { data?: Trip[] }) {
  // defensive: ensure data is always an array
  const rows = Array.isArray(data) ? data : [];

  const table = useReactTable({
    data: rows,
    columns: tripColumns,
    getCoreRowModel: getCoreRowModel(),
    getRowCanExpand: () => true,
  });

  if (rows.length === 0) {
    return (
      <div className="w-full py-6 text-center text-gray-400">
        No trips to show — maybe your schedule is an object and not yet
        normalized.
      </div>
    );
  }

  return (
    <table className="min-w-full border border-zinc-700 text-gray-200">
      <thead className="bg-zinc-900 text-gray-300">
        {table.getHeaderGroups().map((headerGroup) => (
          <tr key={headerGroup.id}>
            {headerGroup.headers.map((header) => (
              <th
                key={header.id}
                className="border border-zinc-700 px-2 py-1 text-left"
              >
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
                <td key={cell.id} className="border border-zinc-700 px-2 py-1">
                  {flexRender(cell.column.columnDef.cell, cell.getContext())}
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
                  <div className="ml-6 mr-6 mt-2 mb-2 border border-zinc-700 rounded bg-zinc-800 shadow-sm">
                    <table className="min-w-[90%] border border-zinc-700 mx-auto text-gray-200">
                      <thead className="bg-zinc-900 text-gray-300">
                        <tr>
                          {arrivalColumns.map((col) => (
                            <th
                              key={
                                col.id?.toString() ||
                                col.accessorKey?.toString()
                              }
                              className="border border-zinc-700 px-2 py-1 text-left"
                            >
                              {typeof col.header === "string"
                                ? col.header
                                : (col.header as any)()}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {row.original.arrivals.map((arr) => (
                          <tr
                            key={arr.stop_id}
                            className="hover:bg-zinc-900 transition"
                          >
                            <td className="border border-zinc-700 px-2 py-1">
                              {arr.stop_id}
                            </td>
                            <td className="border border-zinc-700 px-2 py-1">
                              {arr.arrival_time}
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
