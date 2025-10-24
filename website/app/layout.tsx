import type React from "react"
import type { Metadata } from "next"
import { Analytics } from "@vercel/analytics/next"

const globalStyles = `
*,*::before,*::after{box-sizing:border-box;}
html{font-size:16px;}
body{margin:0;font-family:"Georgia","Times New Roman",serif;background-color:#ffffff;color:#1f2933;-webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale;}
a{text-decoration:underline;text-decoration-thickness:0.08em;text-underline-offset:0.2em;color:inherit;}
img{max-width:100%;height:auto;display:block;}
.min-h-screen{min-height:100vh;}
.bg-white{background-color:#ffffff;}
.bg-\\[\\#f8f7f5\\]{background-color:#f8f7f5;}
.mx-auto{margin-left:auto;margin-right:auto;}
.max-w-\\[750px\\]{max-width:750px;}
.px-8{padding-left:2rem;padding-right:2rem;}
.px-6{padding-left:1.5rem;padding-right:1.5rem;}
.py-16{padding-top:4rem;padding-bottom:4rem;}
.py-6{padding-top:1.5rem;padding-bottom:1.5rem;}
.py-5{padding-top:1.25rem;padding-bottom:1.25rem;}
.mb-12{margin-bottom:3rem;}
.mb-8{margin-bottom:2rem;}
.mb-6{margin-bottom:1.5rem;}
.mb-5{margin-bottom:1.25rem;}
.mb-4{margin-bottom:1rem;}
.mb-3{margin-bottom:0.75rem;}
.mb-2{margin-bottom:0.5rem;}
.mb-1{margin-bottom:0.25rem;}
.mt-12{margin-top:3rem;}
.flex{display:flex;}
.gap-4{gap:1rem;}
.space-y-3>*+*{margin-top:0.75rem;}
.mr-4{margin-right:1rem;}
.text-5xl{font-size:3rem;line-height:1.1;}
.text-2xl{font-size:1.5rem;line-height:2rem;}
.text-xl{font-size:1.25rem;line-height:1.75rem;}
.text-base{font-size:1rem;line-height:1.5rem;}
.text-sm{font-size:0.875rem;line-height:1.25rem;}
.text-xs{font-size:0.75rem;letter-spacing:0.08em;}
.text-\\[17px\\]{font-size:17px;line-height:1.65;}
.font-bold{font-weight:700;}
.font-semibold{font-weight:600;}
.font-sans{font-family:"Inter","Segoe UI",system-ui,-apple-system,sans-serif;}
.italic{font-style:italic;}
.uppercase{text-transform:uppercase;}
.tracking-wide{letter-spacing:0.12em;}
.leading-tight{line-height:1.25;}
.leading-relaxed{line-height:1.75;}
.text-balance{text-wrap:balance;}
.text-pretty{text-wrap:pretty;}
.text-black{color:#000000;}
.text-gray-800{color:#1f2937;}
.text-gray-500{color:#6b7280;}
.text-gray-400{color:#9ca3af;}
.border{border-width:1px;border-style:solid;border-color:#e5e7eb;}
.border-gray-200{border-color:#e5e7eb;}
.rounded-sm{border-radius:0.125rem;}
`;

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
        <style>{globalStyles}</style>
        {children}
        <Analytics />
      </body>
    </html>
  )
}
