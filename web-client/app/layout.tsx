import type { Metadata } from "next";
import "./globals.css";
import { Sidebar } from "@/components/layout/Sidebar";
import { GlobalStatusBar } from "@/components/layout/GlobalStatusBar";
import { CommandPalette } from "@/components/layout/CommandPalette";
import { DataCanary } from "@/components/health/DataCanary";
import { PreFlightChecklist } from "@/components/health/PreFlightChecklist";

export const metadata: Metadata = {
  title: "Polymarket Bot | Industrial Ops",
  description: "High-frequency trading operations dashboard",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark h-full">
      <body className="bg-background text-text-primary antialiased h-full overflow-hidden">
        {/* Layer A: Global Status Bar */}
        <GlobalStatusBar />

        {/* Layer B: Sidebar */}
        <Sidebar />

        {/* Layer C: Main Content */}
        <main className="fixed top-10 left-12 right-0 bottom-0 overflow-y-auto p-0.5 md:p-4">
          {children}
        </main>

        {/* Overlays */}
        <CommandPalette />
        <DataCanary />
        <PreFlightChecklist />
      </body>
    </html>
  );
}
