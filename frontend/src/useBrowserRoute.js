import { useEffect, useState } from "react";

export function useBrowserRoute(validPaths, fallbackPath = "/") {
  const normalize = (pathname) => (validPaths.includes(pathname) ? pathname : fallbackPath);
  const [pathname, setPathname] = useState(() => normalize(window.location.pathname));

  useEffect(() => {
    function handlePopState() {
      setPathname(normalize(window.location.pathname));
    }

    if (window.location.pathname !== pathname) {
      window.history.replaceState({}, "", pathname);
    }

    window.addEventListener("popstate", handlePopState);
    return () => {
      window.removeEventListener("popstate", handlePopState);
    };
  }, [pathname, validPaths]);

  function navigate(nextPath) {
    const resolvedPath = normalize(nextPath);
    if (resolvedPath === window.location.pathname) {
      setPathname(resolvedPath);
      return;
    }
    window.history.pushState({}, "", resolvedPath);
    setPathname(resolvedPath);
  }

  return { pathname, navigate };
}
