"use client";

import React from "react";
import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  useReactTable,
} from "@tanstack/react-table";

// Types
export type Prediction = {
  hour: number;
  predicted_passengers: number;
};

export type RoutePrediction = {
  route_id: number;
  generated_at: string;
  predictions: Prediction[];
};

// Columns for nested predictions
export const predictionColumns: ColumnDef<Prediction>[] = [
  { accessorKey: "hour", header: "Hour" },
  { accessorKey: "predicted_passengers", header: "Predicted Passengers" },
];

// Columns for top-level route prediction
export const routePredictionColumns: ColumnDef<RoutePrediction>[] = [
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
  { accessorKey: "route_id", header: "Route ID" },
  {
    accessorKey: "generated_at",
    header: "Generated At",
    cell: ({ getValue }) => {
      const date = new Date(getValue() as string);
      return date.toLocaleString();
    },
  },
  {
    accessorKey: "predictions",
    header: "Count",
    cell: ({ row }) => row.original.predictions.length,
  },
];

// Main table
export function PredictionTable({ data }: { data: RoutePrediction[] }) {
  const table = useReactTable({
    data,
    columns: routePredictionColumns,
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
                  {flexRender(cell.column.columnDef.cell, cell.getContext())}
                </td>
              ))}
            </tr>

            {/* Expanded row */}
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
                          {predictionColumns.map((col) => (
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
                        {row.original.predictions.map((pred, idx) => (
                          <tr key={idx} className="transition">
                            <td className="border px-2 py-1">{pred.hour}:00</td>
                            <td className="border px-2 py-1">
                              {pred.predicted_passengers}
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
