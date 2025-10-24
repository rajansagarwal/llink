export function ResearchFooter() {
  return (
    <footer className="border-t border-border bg-card mt-16">
      <div className="mx-auto max-w-4xl px-6 py-8">
        <div className="flex flex-col gap-6 md:flex-row md:items-center md:justify-between">
          <div className="space-y-2">
            <p className="text-sm font-medium text-foreground">Citation</p>
            <code className="block rounded bg-muted px-3 py-2 text-xs text-muted-foreground">
              Chen, S., & Rodriguez, J. (2025). Advancing Neural Network Architectures for Real-Time Image Processing.
            </code>
          </div>

          <div className="flex gap-4">
            <a href="#" className="text-sm text-muted-foreground hover:text-accent transition-colors">
              Download PDF
            </a>
            <a href="#" className="text-sm text-muted-foreground hover:text-accent transition-colors">
              View Code
            </a>
            <a href="#" className="text-sm text-muted-foreground hover:text-accent transition-colors">
              Contact Authors
            </a>
          </div>
        </div>
      </div>
    </footer>
  )
}
