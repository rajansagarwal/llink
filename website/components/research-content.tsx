import { Card } from "@/components/ui/card"
import Image from "next/image"

export function ResearchContent() {
  return (
    <main className="mx-auto max-w-4xl px-6 py-12">
      <article className="space-y-12">
        {/* Abstract */}
        <section className="space-y-4">
          <h2 className="text-2xl font-bold text-foreground">Abstract</h2>
          <Card className="bg-muted/50 p-6">
            <p className="leading-relaxed text-card-foreground">
              We present a novel neural network architecture that achieves state-of-the-art performance in real-time
              image processing tasks. Our approach combines attention mechanisms with efficient convolutional layers,
              reducing computational overhead by 40% while maintaining accuracy comparable to existing methods. Through
              extensive experiments on benchmark datasets, we demonstrate significant improvements in both speed and
              quality metrics.
            </p>
          </Card>
        </section>

        {/* Introduction */}
        <section className="space-y-4">
          <h2 className="text-2xl font-bold text-foreground">1. Introduction</h2>
          <div className="space-y-4 leading-relaxed text-foreground">
            <p>
              Real-time image processing has become increasingly critical in applications ranging from autonomous
              vehicles to medical imaging systems. Traditional convolutional neural networks (CNNs) have achieved
              remarkable success in computer vision tasks, but their computational demands often limit deployment in
              resource-constrained environments.
            </p>
            <p>
              Recent advances in attention mechanisms have shown promise in improving model efficiency
              <sup className="text-accent">[1]</sup>. However, existing approaches typically sacrifice either speed or
              accuracy when optimizing for real-time performance. This trade-off presents a significant challenge for
              practical applications where both factors are essential.
            </p>
            <p>
              In this work, we introduce <span className="font-semibold text-primary">EfficientVisionNet</span>, a
              hybrid architecture that leverages selective attention and depth-wise separable convolutions to achieve
              optimal performance across multiple dimensions. Our key contributions include:
            </p>
            <ul className="ml-6 space-y-2 list-disc text-foreground">
              <li>A novel attention module that reduces computational complexity from O(n²) to O(n log n)</li>
              <li>
                An adaptive layer scaling mechanism that dynamically adjusts network depth based on input complexity
              </li>
              <li>Comprehensive benchmarks demonstrating 40% faster inference with comparable accuracy</li>
            </ul>
          </div>
        </section>

        {/* Architecture Diagram */}
        <section className="space-y-4">
          <h2 className="text-2xl font-bold text-foreground">2. Architecture</h2>
          <div className="space-y-6">
            <p className="leading-relaxed text-foreground">
              Our architecture consists of three main components: an efficient feature extraction backbone, a selective
              attention module, and an adaptive output head. Figure 1 illustrates the overall network structure.
            </p>

            <figure className="space-y-3">
              <div className="overflow-hidden rounded-lg border border-border bg-muted">
                <Image
                  src="/neural-network-architecture-diagram-with-layers-an.jpg"
                  alt="EfficientVisionNet Architecture Diagram"
                  width={800}
                  height={400}
                  className="w-full"
                />
              </div>
              <figcaption className="text-sm text-muted-foreground text-center">
                Figure 1: Overview of the EfficientVisionNet architecture showing the feature extraction backbone,
                attention module, and adaptive output head.
              </figcaption>
            </figure>

            <div className="space-y-4 leading-relaxed text-foreground">
              <h3 className="text-xl font-semibold text-foreground">2.1 Feature Extraction Backbone</h3>
              <p>
                The backbone network employs depth-wise separable convolutions to reduce parameter count while
                maintaining representational capacity. Each block consists of:
              </p>
              <Card className="bg-card p-4 font-mono text-sm border-l-4 border-l-accent">
                <code className="text-card-foreground">
                  Conv2D(3×3, depth-wise) → BatchNorm → ReLU → Conv2D(1×1, point-wise) → BatchNorm
                </code>
              </Card>
            </div>
          </div>
        </section>

        {/* Results */}
        <section className="space-y-4">
          <h2 className="text-2xl font-bold text-foreground">3. Experimental Results</h2>
          <div className="space-y-6">
            <p className="leading-relaxed text-foreground">
              We evaluated our approach on three benchmark datasets: ImageNet, COCO, and a custom real-time processing
              dataset. All experiments were conducted using identical hardware configurations to ensure fair comparison.
            </p>

            <div className="overflow-hidden rounded-lg border border-border">
              <table className="w-full text-sm">
                <thead className="bg-muted">
                  <tr className="border-b border-border">
                    <th className="px-4 py-3 text-left font-semibold text-foreground">Model</th>
                    <th className="px-4 py-3 text-right font-semibold text-foreground">Accuracy (%)</th>
                    <th className="px-4 py-3 text-right font-semibold text-foreground">FPS</th>
                    <th className="px-4 py-3 text-right font-semibold text-foreground">Params (M)</th>
                  </tr>
                </thead>
                <tbody className="bg-card">
                  <tr className="border-b border-border">
                    <td className="px-4 py-3 text-card-foreground">ResNet-50</td>
                    <td className="px-4 py-3 text-right text-card-foreground">76.2</td>
                    <td className="px-4 py-3 text-right text-card-foreground">45</td>
                    <td className="px-4 py-3 text-right text-card-foreground">25.6</td>
                  </tr>
                  <tr className="border-b border-border">
                    <td className="px-4 py-3 text-card-foreground">EfficientNet-B0</td>
                    <td className="px-4 py-3 text-right text-card-foreground">77.1</td>
                    <td className="px-4 py-3 text-right text-card-foreground">52</td>
                    <td className="px-4 py-3 text-right text-card-foreground">5.3</td>
                  </tr>
                  <tr className="bg-accent/10">
                    <td className="px-4 py-3 font-semibold text-foreground">EfficientVisionNet (Ours)</td>
                    <td className="px-4 py-3 text-right font-semibold text-foreground">76.8</td>
                    <td className="px-4 py-3 text-right font-semibold text-foreground">73</td>
                    <td className="px-4 py-3 text-right font-semibold text-foreground">4.2</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <p className="text-sm text-muted-foreground">
              Table 1: Performance comparison on ImageNet validation set. Our method achieves 40% higher throughput
              while maintaining competitive accuracy.
            </p>
          </div>
        </section>

        {/* Visualization */}
        <section className="space-y-4">
          <h2 className="text-2xl font-bold text-foreground">4. Qualitative Analysis</h2>
          <div className="space-y-6">
            <p className="leading-relaxed text-foreground">
              Figure 2 shows attention maps generated by our selective attention module. The network successfully
              focuses on relevant image regions while suppressing background noise.
            </p>

            <div className="grid gap-4 md:grid-cols-2">
              <figure className="space-y-2">
                <div className="overflow-hidden rounded-lg border border-border bg-muted">
                  <Image
                    src="/original-input-image-for-neural-network.jpg"
                    alt="Original Input"
                    width={400}
                    height={300}
                    className="w-full"
                  />
                </div>
                <figcaption className="text-sm text-muted-foreground">Original Input</figcaption>
              </figure>

              <figure className="space-y-2">
                <div className="overflow-hidden rounded-lg border border-border bg-muted">
                  <Image
                    src="/attention-heatmap-visualization-overlay.jpg"
                    alt="Attention Map"
                    width={400}
                    height={300}
                    className="w-full"
                  />
                </div>
                <figcaption className="text-sm text-muted-foreground">Attention Heatmap</figcaption>
              </figure>
            </div>
          </div>
        </section>

        {/* Conclusion */}
        <section className="space-y-4">
          <h2 className="text-2xl font-bold text-foreground">5. Conclusion</h2>
          <div className="space-y-4 leading-relaxed text-foreground">
            <p>
              We have presented EfficientVisionNet, a novel architecture for real-time image processing that achieves
              significant improvements in computational efficiency without sacrificing accuracy. Our selective attention
              mechanism and adaptive layer scaling enable deployment in resource-constrained environments while
              maintaining state-of-the-art performance.
            </p>
            <p>
              Future work will explore extensions to video processing tasks and investigate the application of our
              approach to other domains such as natural language processing and multimodal learning.
            </p>
          </div>
        </section>

        {/* References */}
        <section className="space-y-4 border-t border-border pt-8">
          <h2 className="text-2xl font-bold text-foreground">References</h2>
          <ol className="space-y-3 text-sm leading-relaxed text-foreground">
            <li className="flex gap-3">
              <span className="text-muted-foreground">[1]</span>
              <span>
                Vaswani, A., et al. (2017). "Attention is All You Need."{" "}
                <em>Advances in Neural Information Processing Systems</em>, 30.
              </span>
            </li>
            <li className="flex gap-3">
              <span className="text-muted-foreground">[2]</span>
              <span>
                He, K., et al. (2016). "Deep Residual Learning for Image Recognition."{" "}
                <em>Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition</em>, 770-778.
              </span>
            </li>
            <li className="flex gap-3">
              <span className="text-muted-foreground">[3]</span>
              <span>
                Tan, M., & Le, Q. (2019). "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks."{" "}
                <em>International Conference on Machine Learning</em>, 6105-6114.
              </span>
            </li>
          </ol>
        </section>
      </article>
    </main>
  )
}
