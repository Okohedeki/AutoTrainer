import React from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const plans = [
  ["Starter", "$12"],
  ["Studio", "$29"],
  ["Agency", "$79"],
];

function App() {
  return (
    <main>
      <h1>Plans for growing teams</h1>
      <section className="pricing-grid" aria-label="Pricing plans">
        {plans.map(([name, price]) => (
          <article className="pricing-card" key={name}>
            <h2>{name}</h2>
            <p className="price">{price}<span>/month</span></p>
            <button type="button">Choose {name}</button>
          </article>
        ))}
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
