// The Newsren — small progressive enhancements on top of server-rendered pages.
// Everything works without JavaScript; htmx makes the intelligence tools load in place.

(function () {
  "use strict";

  // Tool tabs: mark the active tab and keep aria-selected in sync.
  document.addEventListener("click", function (event) {
    const tab = event.target.closest("[data-tool-tab]");
    if (!tab) return;
    const group = tab.closest("[role=tablist]");
    if (!group) return;
    group.querySelectorAll("[data-tool-tab]").forEach(function (t) {
      const active = t === tab;
      t.classList.toggle("is-active", active);
      t.setAttribute("aria-selected", active ? "true" : "false");
    });
  });

  // Suggested prompts fill the nearest question box.
  document.addEventListener("click", function (event) {
    const chip = event.target.closest("[data-prompt]");
    if (!chip) return;
    const form = chip.closest("form") || document.querySelector("[data-ask-form]");
    const input = form && form.querySelector("textarea, input[name=question]");
    if (input) {
      event.preventDefault();
      input.value = chip.getAttribute("data-prompt");
      input.focus();
    }
  });

  // Quiz feedback without a round trip (the server still records the attempt).
  document.addEventListener("change", function (event) {
    const option = event.target.closest("[data-quiz-option]");
    if (!option) return;
    const question = option.closest("[data-quiz-question]");
    const result = question && question.querySelector("[data-quiz-result]");
    if (!result) return;
    const correct = option.value === question.getAttribute("data-answer");
    result.textContent = correct ? "Correct. " + (question.getAttribute("data-explanation") || "") : "Not quite — try again or read the explainer.";
    result.className = "quiz-result " + (correct ? "correct" : "incorrect");
  });

  // Surface htmx errors in place instead of failing silently.
  document.addEventListener("htmx:responseError", function (event) {
    const target = event.detail.target;
    if (target) {
      target.innerHTML = '<p class="flash flash-error">That tool could not load right now. Please try again.</p>';
    }
  });
})();
