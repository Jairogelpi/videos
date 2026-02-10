import type { Metadata } from "next";
import { Inter, Outfit, Montserrat, Syne, Anton, Bebas_Neue, Unbounded, Space_Grotesk, Playfair_Display } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });
const outfit = Outfit({ subsets: ["latin"], variable: "--font-outfit" });
const montserrat = Montserrat({ subsets: ["latin"], variable: "--font-montserrat" });
const syne = Syne({ subsets: ["latin"], variable: "--font-syne" });
const anton = Anton({ weight: '400', subsets: ["latin"], variable: "--font-anton" });
const bebas = Bebas_Neue({ weight: '400', subsets: ["latin"], variable: "--font-bebas-neue" });
const unbounded = Unbounded({ subsets: ["latin"], variable: "--font-unbounded" });
const spaceGrotesk = Space_Grotesk({ subsets: ["latin"], variable: "--font-space-grotesk" });
const playfair = Playfair_Display({ subsets: ["latin"], variable: "--font-playfair-display" });

export const metadata: Metadata = {
  title: "TOHJO Studio | Cinematic Sync Alpha",
  description: "Elite Music Video Production with Local LTX-Video",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="es" className="dark">
      <body className={`${inter.variable} ${outfit.variable} ${montserrat.variable} ${syne.variable} ${anton.variable} ${bebas.variable} ${unbounded.variable} ${spaceGrotesk.variable} ${playfair.variable} font-sans bg-black text-white antialiased selection:bg-indigo-500/30`}>
        {children}
      </body>
    </html>
  );
}
