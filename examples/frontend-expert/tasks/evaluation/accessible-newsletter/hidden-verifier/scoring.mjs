const escapeRegExp = (value) => value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

function namedChecks(values) {
  // Preserve every named signal in the report as well as its normalized rate.
  // This keeps rewards auditable instead of collapsing them into one opaque bit.
  const entries = Object.entries(values).map(([name, passed]) => [name, Boolean(passed)]);
  return {
    checks: Object.fromEntries(entries),
    rate: entries.length === 0 ? 0 : entries.filter(([, passed]) => passed).length / entries.length,
  };
}

function attribute(tag, name) {
  return tag.match(new RegExp(`\\b${escapeRegExp(name)}\\s*=\\s*["']([^"']+)["']`, "i"))?.[1] ?? "";
}

function mediaBlocks(css) {
  // The verifier inspects source text and never imports or executes policy code.
  // This small brace walker is sufficient for the fixture's flat media queries.
  const blocks = [];
  const start = /@media\s*\(([^)]*)\)\s*\{/gi;
  for (let match = start.exec(css); match; match = start.exec(css)) {
    let depth = 1;
    let cursor = start.lastIndex;
    while (cursor < css.length && depth > 0) {
      if (css[cursor] === "{") depth += 1;
      if (css[cursor] === "}") depth -= 1;
      cursor += 1;
    }
    blocks.push({ condition: match[1], body: css.slice(start.lastIndex, cursor - 1) });
    start.lastIndex = cursor;
  }
  return blocks;
}

function maxWidthPixels(condition) {
  const match = condition.match(/max-width\s*:\s*([\d.]+)(px|r?em)/i);
  if (!match) return Number.POSITIVE_INFINITY;
  return Number(match[1]) * (match[2].toLowerCase() === "px" ? 1 : 16);
}

function selectorBody(css, selector) {
  return [...css.matchAll(new RegExp(`${escapeRegExp(selector)}\\s*\\{([^{}]*)\\}`, "gi"))]
    .map((match) => match[1])
    .join("\n");
}

function stacks(css, selector) {
  const body = selectorBody(css, selector);
  return /grid-template-columns\s*:\s*(?:minmax\(\s*0\s*,\s*)?1fr\s*\)?\s*;/i.test(body)
    || /flex-direction\s*:\s*column\b/i.test(body)
    || /display\s*:\s*block\b/i.test(body);
}

function hasTouchTarget(css) {
  const bodies = [...css.matchAll(/[^{}]*button[^{}]*\{([^{}]*)\}/gi)].map((match) => match[1]);
  return bodies.some((body) => {
    const value = body.match(/(?:min-height|min-block-size)\s*:\s*([^;]+)/i)?.[1] ?? "";
    const lengths = [...value.matchAll(/([\d.]+)(px|r?em)/gi)].map((match) =>
      Number(match[1]) * (match[2].toLowerCase() === "px" ? 1 : 16),
    );
    return lengths.some((length) => length >= 44);
  });
}

function hasVisibleFocus(css) {
  return /:focus-visible[^{}]*\{[^{}]*(?:outline\s*:\s*(?!none\b)[^;]+|box-shadow\s*:\s*(?!none\b)[^;]+)/is.test(css);
}

export function scoreSources(jsx, css) {
  // These independent groups map directly to the V1 reward components. Public
  // regressions protect the brief; hidden checks measure the requested repair.
  const emailInput = (jsx.match(/<input\b[^>]*>/gis) ?? []).find((tag) => /type\s*=\s*["']email["']/i.test(tag)) ?? "";
  const inputId = attribute(emailInput, "id");
  const associatedLabel = Boolean(inputId) && new RegExp(
    `<label\\b[^>]*htmlFor\\s*=\\s*["']${escapeRegExp(inputId)}["'][^>]*>[\\s\\S]*?Email address[\\s\\S]*?<\\/label>`,
    "i",
  ).test(jsx);
  const mobileCss = mediaBlocks(css)
    .filter(({ condition }) => maxWidthPixels(condition) <= 800)
    .map(({ body }) => body)
    .join("\n");
  const focusVisible = hasVisibleFocus(css);

  const accessibility = namedChecks({
    associated_email_label: associatedLabel,
    email_autocomplete_hint: /autoComplete\s*=\s*["']email["']/i.test(emailInput),
    email_type_and_required: /type\s*=\s*["']email["']/i.test(emailInput) && /\brequired\b/i.test(emailInput),
    named_signup_form: /<form\b[^>]*(?:aria-label|aria-labelledby)\s*=/i.test(jsx),
    announced_feedback: /<(?:p|div)\b[^>]*(?:role\s*=\s*["']status["']|aria-live\s*=\s*["'](?:polite|assertive)["'])[^>]*>[\s\S]*?\{message\}/i.test(jsx),
    visible_focus_indicator: focusVisible,
    semantic_landmarks: ["header", "nav", "main", "section", "aside", "footer"].every((tag) => new RegExp(`<${tag}\\b`, "i").test(jsx)),
    explicit_submit_action: /<button\b[^>]*type\s*=\s*["']submit["']/i.test(jsx),
  });

  const responsive = namedChecks({
    narrow_breakpoint: mobileCss.length > 0,
    hero_stacks: stacks(mobileCss, ".hero"),
    signup_controls_stack: stacks(mobileCss, ".form-controls"),
    issue_cards_stack: stacks(mobileCss, ".issue-grid"),
    minimum_action_target: hasTouchTarget(css),
    fluid_email_control: /\.form-controls\s+input\s*\{[^}]*width\s*:\s*100%/is.test(css),
    predictable_box_sizing: /\*\s*\{[^}]*box-sizing\s*:\s*border-box/is.test(css),
  });

  const design = namedChecks({
    design_tokens: (css.match(/--[\w-]+\s*:/g) ?? []).length >= 8,
    token_usage: (css.match(/var\(--[\w-]+\)/g) ?? []).length >= 12,
    readable_body_leading: /body\s*\{[^}]*line-height\s*:\s*1\.[4-8]/is.test(css),
    fluid_type_scale: /font-size\s*:\s*clamp\(/i.test(css),
    reduced_motion_option: /prefers-reduced-motion\s*:\s*reduce/i.test(css),
    intentional_focus_treatment: focusVisible,
  });

  const quality = namedChecks({
    no_important_overrides: !/!important/i.test(css),
    no_inline_styles: !/\bstyle\s*=\s*\{/i.test(jsx),
    no_unsafe_html: !/dangerouslySetInnerHTML/.test(jsx),
    no_network_submission: !/\b(?:fetch|XMLHttpRequest|axios)\b/.test(jsx),
    stable_collection_key: /key\s*=\s*\{issue\.number\}/.test(jsx),
    typed_submit_event: /FormEvent<HTMLFormElement>/.test(jsx),
    local_state_feedback: /event\.preventDefault\(\)/.test(jsx) && /setMessage\(/.test(jsx),
    no_unreplaced_focus_suppression: !/outline\s*:\s*none/i.test(css) || focusVisible,
  });

  const regression = namedChecks({
    publication_identity: jsx.includes("Signal &amp; Story"),
    hero_copy: jsx.includes("Make space for ideas worth keeping.") && jsx.includes("A calmer way to follow the ideas shaping creative work."),
    signup_copy: jsx.includes("The next note arrives Thursday."),
    archive_titles: ["The patient city", "Tools that leave room", "A practice of noticing"].every((title) => jsx.includes(title)),
    exactly_three_issues: (jsx.match(/number:\s*["']Issue \d+["']/g) ?? []).length === 3,
    local_submit_behavior: /event\.preventDefault\(\)/.test(jsx) && /new FormData\(event\.currentTarget\)/.test(jsx) && /setMessage\(/.test(jsx),
    desktop_hero_composition: /\.hero\s*\{[^}]*grid-template-columns\s*:\s*minmax\(0,\s*1\.15fr\)\s+minmax\(20rem,\s*0\.85fr\)/is.test(css),
    desktop_archive_composition: /\.issue-grid\s*\{[^}]*grid-template-columns\s*:\s*repeat\(3,\s*minmax\(12rem,\s*1fr\)\)/is.test(css),
  });

  const task = namedChecks({
    ...accessibility.checks,
    narrow_breakpoint: responsive.checks.narrow_breakpoint,
    hero_stacks: responsive.checks.hero_stacks,
    signup_controls_stack: responsive.checks.signup_controls_stack,
    issue_cards_stack: responsive.checks.issue_cards_stack,
    minimum_action_target: responsive.checks.minimum_action_target,
  });

  return {
    regression_pass_rate: regression.rate,
    task_pass_rate: task.rate,
    responsive_pass_rate: responsive.rate,
    design_rule_pass_rate: design.rate,
    code_quality_pass_rate: quality.rate,
    diagnostics: {
      accessibility: accessibility.checks,
      responsive: responsive.checks,
      design: design.checks,
      code_quality: quality.checks,
      regression: regression.checks,
    },
  };
}
