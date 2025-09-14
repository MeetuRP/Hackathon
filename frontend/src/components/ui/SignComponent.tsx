"use client";

import { useEffect, useState } from "react";
import Cookies from "js-cookie";
import Link from "next/link";
import { DropdownMenuItem } from "@radix-ui/react-dropdown-menu";

export default function Signcomponent() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  useEffect(() => {
    const email = Cookies.get("email"); // 👈 check cookie
    setIsLoggedIn(!!email);
  }, []);

  const handleSignOut = () => {
    Cookies.remove("email"); // 👈 remove cookie
    setIsLoggedIn(false);
  };

  return (
    <DropdownMenuItem>
      {isLoggedIn ? (
        <Link href={"/login"}>
          <button onClick={handleSignOut}>Sign out</button>
        </Link>
      ) : (
        <Link href={"/register"}>
          <button onClick={handleSignOut}>Sign in</button>
        </Link>
      )}
    </DropdownMenuItem>
  );
}
