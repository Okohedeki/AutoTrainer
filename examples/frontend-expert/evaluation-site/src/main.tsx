import type { FormEvent } from "react";
import { useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

// Static archive data keeps the held-out starting state reproducible and lets
// public regressions protect exact editorial content without network fixtures.
const issues = [
  {
    number: "Issue 08",
    title: "The patient city",
    summary: "What public benches, late buses, and small rituals teach us about useful pace.",
    readTime: "6 minute read",
  },
  {
    number: "Issue 07",
    title: "Tools that leave room",
    summary: "A field guide to choosing software that supports attention instead of competing for it.",
    readTime: "5 minute read",
  },
  {
    number: "Issue 06",
    title: "A practice of noticing",
    summary: "Three quiet prompts for finding the detail that makes an ordinary story memorable.",
    readTime: "4 minute read",
  },
] as const;

function App() {
  const [message, setMessage] = useState("");

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    // Subscription behavior stays local in V1 so an evaluation episode never
    // needs credentials or outbound network access.
    const data = new FormData(event.currentTarget);
    const email = String(data.get("email") ?? "").trim();
    setMessage(email ? `You're subscribed—look for a note at ${email}.` : "Enter your email to join.");
  }

  return (
    <>
      <header className="site-header">
        <a className="wordmark" href="#top" aria-label="Signal and Story home">
          <span aria-hidden="true">S/S</span>
          Signal &amp; Story
        </a>
        <nav aria-label="Primary navigation">
          <a href="#latest">Latest issues</a>
          <a href="#about">About</a>
        </nav>
      </header>

      <main id="top">
        <section className="hero" aria-labelledby="hero-title">
          <div className="hero-copy">
            <p className="eyebrow">A weekly field note for curious people</p>
            <h1 id="hero-title">Make space for ideas worth keeping.</h1>
            <p className="intro">
              A calmer way to follow the ideas shaping creative work. One considered essay,
              three useful links, and a question to carry into the week.
            </p>
            <ul className="proof-list" aria-label="Newsletter details">
              <li>Every Thursday</li>
              <li>Under eight minutes</li>
              <li>No tracking pixels</li>
            </ul>
          </div>

          <aside className="signup-panel" aria-labelledby="signup-title">
            <p className="panel-kicker">Join 2,400 thoughtful readers</p>
            <h2 id="signup-title">The next note arrives Thursday.</h2>
            <p>Free to read, easy to leave. Start with Issue 08 in your welcome email.</p>
            <form className="signup-form" onSubmit={handleSubmit}>
              <span className="form-label">Email address</span>
              <div className="form-controls">
                <input
                  id="subscriber-email"
                  name="email"
                  type="email"
                  placeholder="you@example.com"
                  required
                />
                <button type="submit">Join free</button>
              </div>
            </form>
            {message && <p className="form-status">{message}</p>}
            <p className="privacy-note">No spam. Your address stays with Signal &amp; Story.</p>
          </aside>
        </section>

        <section className="latest" id="latest" aria-labelledby="latest-title">
          <div className="section-heading">
            <div>
              <p className="eyebrow">From the archive</p>
              <h2 id="latest-title">Three notes to begin with</h2>
            </div>
            <a href="#top">Get the next issue <span aria-hidden="true">→</span></a>
          </div>

          <div className="issue-grid">
            {issues.map((issue) => (
              <article className="issue-card" key={issue.number}>
                <p className="issue-number">{issue.number}</p>
                <h3>{issue.title}</h3>
                <p>{issue.summary}</p>
                <p className="read-time">{issue.readTime}</p>
              </article>
            ))}
          </div>
        </section>

        <section className="about" id="about" aria-labelledby="about-title">
          <p className="eyebrow">Behind the letter</p>
          <h2 id="about-title">Written slowly. Sent with purpose.</h2>
          <p>
            Signal &amp; Story is edited by Mara Bell, an independent researcher exploring how
            design, place, and technology shape the way we pay attention.
          </p>
        </section>
      </main>

      <footer>
        <p>© 2026 Signal &amp; Story</p>
        <a href="mailto:hello@example.test">hello@example.test</a>
      </footer>
    </>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
