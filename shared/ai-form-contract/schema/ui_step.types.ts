/**
 * UI Step types for the AI form flow (the "Real Schema").
 *
 * Treat `ui_step.schema.json` as the canonical source of truth; this file is a
 * convenient TS mirror for UI consumption.
 */

/**
 * Step types supported by the shared UI schema.
 *
 * Note: this includes legacy aliases that the UI still accepts (e.g. `text`, `choice`, `slider`, `upload`),
 * so the frontend can consume both LLM-generated steps and deterministic structural steps.
 */
export type UIStepType =
  | "intro"
  | "text_input"
  | "text"
  | "multiple_choice"
  | "choice"
  | "segmented_choice"
  | "chips_multi"
  | "yes_no"
  | "image_choice_grid"
  | "rating"
  | "slider"
  | "range_slider"
  | "file_upload"
  | "upload"
  | "file_picker"
  | "budget_cards"
  | "date_picker"
  | "color_picker"
  | "searchable_select"
  | "lead_capture"
  | "pricing"
  | "confirmation"
  | "designer"
  | "composite";

/**
 * Option definition. This is intentionally "rich": options can carry descriptions/icons/images
 * so the UI can render image grids, cards, etc.
 */
export type UIOption = {
  label: string;
  value: string;
  description?: string | null;
  icon?: string | null;
  imageUrl?: string | null;
};

/**
 * A fuller "step blueprint" that can describe subcomponents, validation, and presentation.
 *
 * This is optional and fully backwards-compatible: existing steps can omit it.
 */
export type UIStepComponent = {
  /** Component kind (e.g. "headline", "helper", "options", "slider", "notice") */
  type: string;
  /** Unique key within the step (useful for analytics + stable React keys) */
  key?: string;
  /** Optional text payload */
  text?: string | null;
  /** Whether this component is required for a "valid" step */
  required?: boolean | null;
  /** Arbitrary props for the component */
  props?: Record<string, any> | null;
};

export type UIStepBlueprint = {
  /** Declarative list of subcomponents that make up the step UI */
  components?: UIStepComponent[] | null;
  /** Declarative validation rules (frontend + backend can enforce) */
  validation?: Record<string, any> | null;
  /** Presentation hints for the UI layer (labels, auto-advance, etc.) */
  presentation?: {
    continue_label?: string | null;
    auto_advance?: boolean | null;
  } | null;
  /** Optional hint for the LLM about how to generate the step content */
  ai_hint?: string | null;
};

export type UIStepBase = {
  id: string;
  type: UIStepType;
  question: string;
  humanism?: string | null;
  visual_hint?: string | null;
  required?: boolean | null;
  metric_gain?: number | null;
  blueprint?: UIStepBlueprint | null;
};

export type TextInputUI = UIStepBase & {
  type: "text_input" | "text";
  max_length?: number | null;
  placeholder?: string | null;
  multiline?: boolean | null;
};

export type MultipleChoiceUI = UIStepBase & {
  type:
    | "multiple_choice"
    | "choice"
    | "segmented_choice"
    | "chips_multi"
    | "yes_no"
    | "image_choice_grid";
  options: UIOption[];
  multi_select?: boolean | null;
  max_selections?: number | null;
  variant?: "list" | "grid" | "compact" | "cards" | null;
  columns?: number | null;
};

export type RatingUI = UIStepBase & {
  type: "rating" | "slider" | "range_slider";
  scale_min: number;
  scale_max: number;
  step?: number | null;
  anchors?: Record<string, string> | null;
};

export type FileUploadUI = UIStepBase & {
  type: "file_upload" | "upload" | "file_picker";
  allowed_file_types?: string[] | null;
  max_size_mb?: number | null;
  upload_role?: string | null;
  max_files?: number | null;
  allow_skip?: boolean | null;
};

export type BudgetCardsUI = UIStepBase & {
  type: "budget_cards";
  ranges: Array<Record<string, any>>;
  allow_custom?: boolean | null;
  custom_min?: number | null;
  custom_max?: number | null;
  currency_code?: string | null;
};

export type IntroUI = UIStepBase & {
  type: "intro";
  brand?: string | null;
  bullets?: string[] | null;
};

export type DatePickerUI = UIStepBase & {
  type: "date_picker";
  min_date?: string | null;
  max_date?: string | null;
};

export type ColorPickerUI = UIStepBase & {
  type: "color_picker";
  colors?: string[] | null;
};

export type SearchableSelectUI = UIStepBase & {
  type: "searchable_select";
  options: UIOption[];
  multi_select?: boolean | null;
  max_selections?: number | null;
  search_placeholder?: string | null;
};

export type LeadCaptureUI = UIStepBase & {
  type: "lead_capture";
  required_inputs?: Array<"email" | "phone" | "name"> | null;
  require_terms?: boolean | null;
  compact?: boolean | null;
};

export type PricingUI = UIStepBase & {
  type: "pricing";
  pricing_breakdown?: Array<{ label: string; amount: number; unit?: string | null }> | null;
  total_amount?: number | null;
  currency_code?: string | null;
  call_to_action?: string | null;
};

export type ConfirmationUI = UIStepBase & {
  type: "confirmation";
  summary_text?: string | null;
  confirmation_message?: string | null;
};

export type DesignerUI = UIStepBase & {
  type: "designer";
  allow_refinements?: boolean | null;
};

export type CompositeUI = UIStepBase & {
  type: "composite";
  blocks: Array<Record<string, any>>;
};

export type UIStep =
  | IntroUI
  | TextInputUI
  | MultipleChoiceUI
  | RatingUI
  | FileUploadUI
  | BudgetCardsUI
  | DatePickerUI
  | ColorPickerUI
  | SearchableSelectUI
  | LeadCaptureUI
  | PricingUI
  | ConfirmationUI
  | DesignerUI
  | CompositeUI;


