import Login from "@/components/Login";
import React from "react";

const page = () => {
  return (
    <div className="w-[80dvw] h-full flex flex-col items-center justify-center  gap-3">
      <div className="text-2xl text-foreground font-semibold">Login</div>
      <Login />
    </div>
  );
};

export default page;
