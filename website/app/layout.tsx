import type React from "react"
import type { Metadata } from "next"
import { Analytics } from "@vercel/analytics/next"

export const metadata: Metadata = {
  title: "LLINK: Cross-Lingual Alignment via Encoder Injection",
  description:
    "LLINK treats low-resource languages as a modality for instruction-tuned LLMs, aligning multilingual encoder vectors to frozen decoders via soft slots.",
  generator: "v0.app",
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <body
        style={{
          margin: 0,
          fontFamily: "\"Georgia\", \"Times New Roman\", serif",
          backgroundColor: "#ffffff",
          color: "#1f2933",
          WebkitFontSmoothing: "antialiased",
          MozOsxFontSmoothing: "grayscale",
        }}
      >
        {children}
        <Analytics />
      </body>
    </html>
  )
}
