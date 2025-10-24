import Link from "next/link"

export function ResearchHeader() {
  return (
    <header className="border-b border-border bg-card">
      <div className="mx-auto max-w-4xl px-6 py-8">
        <nav className="mb-8 flex items-center justify-between">
          <Link href="/" className="text-sm font-medium text-foreground hover:text-accent transition-colors">
            Research Lab
          </Link>
          <div className="flex items-center gap-6">
            <Link href="/papers" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
              Papers
            </Link>
            <Link href="/about" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
              About
            </Link>
          </div>
        </nav>

        <div className="space-y-6">
          <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
            <span className="rounded-full bg-secondary px-3 py-1 text-secondary-foreground">Machine Learning</span>
            <span className="rounded-full bg-secondary px-3 py-1 text-secondary-foreground">Computer Vision</span>
            <span>Published: October 2025</span>
          </div>

          <h1 className="text-4xl font-bold tracking-tight text-balance text-foreground md:text-5xl">
            Advancing Neural Network Architectures for Real-Time Image Processing
          </h1>

          <div className="flex flex-wrap items-center gap-4 text-sm">
            <div className="flex items-center gap-2">
              <div className="h-10 w-10 rounded-full bg-muted" />
              <div>
                <div className="font-medium text-foreground">Dr. Sarah Chen</div>
                <div className="text-muted-foreground">MIT Computer Science</div>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <div className="h-10 w-10 rounded-full bg-muted" />
              <div>
                <div className="font-medium text-foreground">Dr. James Rodriguez</div>
                <div className="text-muted-foreground">Stanford AI Lab</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </header>
  )
}
