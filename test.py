from LeanRAG import Module, dataset_from_json

# Mathlib = Module("Mathlib", lean_dir="/Users/trowney/Documents/GitHub/test_project")


dataset = dataset_from_json("dataset_minictx_theorems_only.json", lean_dir="/Users/trowney/Documents/GitHub/test_project")

for module in dataset.all_files():
    for decl in module.declarations():
        print(decl.name)
        # print(decl.src)
        print(decl.informal_statement)
        print("-"*50)
    print("="*50 + "\n\n")


# for decl in dataset.Mathlib.Geometry.Manifold.IntegralCurve.Basic.declarations():
#     if decl.name == "eventually_hasDerivAt":
#         print(decl.name)
#         print(decl.src)

# raw = """
#
#
# lemma test := by rfl
#
# lemma IsIntegralCurveOn.hasDerivAt (hγ : IsIntegralCurveOn γ v s) {t : ℝ} (ht : t ∈ s)
#     (hsrc : γ t ∈ (extChartAt I (γ t₀)).source) :
#     HasDerivAt ((extChartAt I (γ t₀)) ∘ γ)
#       (tangentCoordChange I (γ t) (γ t₀) (γ t) (v (γ t))) t := by
#   -- turn `HasDerivAt` into comp of `HasMFDerivAt`
#   have hsrc := extChartAt_source I (γ t₀) ▸ hsrc
#   rw [hasDerivAt_iff_hasFDerivAt, ← hasMFDerivAt_iff_hasFDerivAt]
#   apply (HasMFDerivAt.comp t
#     (hasMFDerivAt_extChartAt (I := I) hsrc) (hγ _ ht)).congr_mfderiv
#   rw [ContinuousLinearMap.ext_iff]
#   intro a
#   rw [ContinuousLinearMap.comp_apply, ContinuousLinearMap.smulRight_apply, map_smul,
#     ← ContinuousLinearMap.one_apply (R₁ := ℝ) a, ← ContinuousLinearMap.smulRight_apply,
#     mfderiv_chartAt_eq_tangentCoordChange hsrc]
#   rfl
# """
# print(get_decls_from_plaintext(raw, "Mathlib.Geometry.Manifold.IntegralCurve.Basic"))