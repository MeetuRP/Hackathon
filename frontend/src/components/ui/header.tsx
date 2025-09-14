"use client";
import { TypewriterEffectSmooth } from "../ui/typewriter-effect";
export function Header() {
  const words = [
    {
      text: "Ai",
    },
    {
      text: "Powered",
    },
    {
      text: "Bus Route",
    },
    {
      text: "Optimization",
    },
    {
      text: "System.",
      className: "text-blue-500 dark:text-blue-500",
    },
  ];
  return (
    <div className="flex flex-col items-center justify-center">
      <TypewriterEffectSmooth words={words} />
    </div>
  );
}
