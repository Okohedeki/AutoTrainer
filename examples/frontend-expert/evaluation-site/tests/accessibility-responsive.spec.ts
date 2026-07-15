import { expect, test } from "@playwright/test";

test("signup field has a persistent label and announces local feedback", async ({ page }) => {
  await page.goto("/");
  const email = page.locator("#subscriber-email");
  await expect(email).toBeVisible();
  const hasPersistentLabel = await email.evaluate((element: HTMLInputElement) =>
    Array.from(element.labels ?? []).some((label) => label.textContent?.trim() === "Email address"),
  );
  expect(hasPersistentLabel).toBe(true);
  await email.fill("reader@example.com");
  await page.getByRole("button", { name: "Join free" }).click();
  await expect(page.getByRole("status")).toContainText("reader@example.com");
});
test("email field exposes a visible keyboard focus indicator", async ({ page }) => {
  await page.goto("/");
  const email = page.locator("#subscriber-email");
  await email.focus();
  const hasVisibleFocus = await email.evaluate((element) => {
    const styles = getComputedStyle(element);
    const outlined = styles.outlineStyle !== "none" && Number.parseFloat(styles.outlineWidth) > 0;
    const shadowed = styles.boxShadow !== "none" && styles.boxShadow.trim() !== "";
    return outlined || shadowed;
  });
  expect(hasVisibleFocus).toBe(true);
});

test("narrow layout stacks content and avoids horizontal overflow", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  const dimensions = await page.evaluate(() => ({
    documentWidth: document.documentElement.scrollWidth,
    viewportWidth: window.innerWidth,
  }));
  expect(dimensions.documentWidth).toBeLessThanOrEqual(dimensions.viewportWidth);

  const heroCopy = await page.locator(".hero-copy").boundingBox();
  const signup = await page.locator(".signup-panel").boundingBox();
  const email = await page.locator("#subscriber-email").boundingBox();
  const button = await page.getByRole("button", { name: "Join free" }).boundingBox();
  expect(heroCopy).not.toBeNull();
  expect(signup).not.toBeNull();
  expect(email).not.toBeNull();
  expect(button).not.toBeNull();
  expect(signup!.y).toBeGreaterThanOrEqual(heroCopy!.y + heroCopy!.height - 1);
  expect(button!.y).toBeGreaterThanOrEqual(email!.y + email!.height - 1);
  expect(button!.height).toBeGreaterThanOrEqual(44);

  const cards = page.locator(".issue-card");
  await expect(cards).toHaveCount(3);
  const first = await cards.nth(0).boundingBox();
  const second = await cards.nth(1).boundingBox();
  expect(first).not.toBeNull();
  expect(second).not.toBeNull();
  expect(second!.y).toBeGreaterThan(first!.y + first!.height - 1);
});

test("wide layout keeps the two-part hero and three-card archive", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 });
  await page.goto("/");
  const heroColumns = await page.locator(".hero").evaluate((element) =>
    getComputedStyle(element).gridTemplateColumns.split(" ").filter(Boolean),
  );
  const archiveColumns = await page.locator(".issue-grid").evaluate((element) =>
    getComputedStyle(element).gridTemplateColumns.split(" ").filter(Boolean),
  );
  expect(heroColumns).toHaveLength(2);
  expect(archiveColumns).toHaveLength(3);
});
