import { expect, test } from "@playwright/test";

test("pricing cards use one readable mobile column", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");

  const grid = page.locator(".pricing-grid");
  await expect(grid).toBeVisible();
  const columns = await grid.evaluate((element) =>
    getComputedStyle(element).gridTemplateColumns.split(" ").filter(Boolean),
  );
  expect(columns).toHaveLength(1);
  await expect(page.getByRole("button", { name: /Choose/ })).toHaveCount(3);
});

test("pricing cards preserve three desktop columns", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 });
  await page.goto("/");

  const columns = await page.locator(".pricing-grid").evaluate((element) =>
    getComputedStyle(element).gridTemplateColumns.split(" ").filter(Boolean),
  );
  expect(columns).toHaveLength(3);
});
