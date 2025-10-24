import type React from "react"
import type { Metadata } from "next"
import { Crimson_Text, Inter } from "next/font/google"
import { Analytics } from "@vercel/analytics/next"
import "./globals.css"

const crimsonText = Crimson_Text({
  subsets: ["latin"],
  weight: ["400", "600", "700"],
  variable: "--font-serif",
})

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-sans",
})

export const metadata: Metadata = {
  title: "MoMoE: Memory-optimized Mixture of Experts | Tilde",
  description:
    "An MoE implementation built with fused Triton kernels, optimized memory packing, and a configurable backward pass for memory-compute trade-offs.",
  generator: "v0.app",
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <body className={`${crimsonText.variable} ${inter.variable} font-serif antialiased`}>
        {children}
        <Analytics />
      </body>
    </html>
  )
}
