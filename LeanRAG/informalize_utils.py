def make_prompt(declaration : str, include_context=False):
    context = ""
    context_prompt = "You will additionally be given the (formal) context/dependencies of the theorem (such as referenced lemmas), which you should use to inform your informalization of the theorem and proof. This context will be wrapped in <CONTEXT>...</CONTEXT> tags."
    if include_context:
        # deps = thm.get("C1_dependencies", [])
        deps = []
        dep_texts = "\n\n".join(d.get("content", "").strip() for d in deps)
        if dep_texts:
            context = f"<CONTEXT>\n{dep_texts}\n</CONTEXT>\n"

            # TODO: Add context example?

    # theorem_text = thm.get("id", {}).get("content", "").strip()
    theorem_text = declaration

    prompt = f"""You are an expert in mathematics and formal theorem proving. Your task is to provide an informal statement and informal step-by-step proof for the following Lean4 formal theorem and proof.

    Namely, you will be given a formal theorem and proof in Lean4 (wrapped in <FORMAL>...</FORMAL> tags), and you need to (1) provide an informal statement of the theorem in natural language (wrap this part of your output in <STATEMENT>...</STATEMENT> tags), 
    and (2) provide an informal proof of the theorem in natural language by translating the formal proof tactic by tactic to produce a step-by-step aligned human-readable proof (wrapped in <PROOF>...</PROOF> tags). Do not skip any steps or omit and details in the proof.
    {context_prompt if include_context else ""}

    Consider the following simple example (wrapped in <EXAMPLE>...</EXAMPLE> tags):
    <EXAMPLE>

    Input:
    <FORMAL>

    theorem primes_infinite : ∀ n, ∃ p > n, Nat.Prime p := by
      intro n
      have : 2 ≤ Nat.factorial (n + 1) + 1 := by
        apply Nat.succ_le_succ
        exact Nat.succ_le_of_lt (Nat.factorial_pos _)
      rcases exists_prime_factor this with ⟨p, pp, pdvd⟩
      refine ⟨p, ?_, pp⟩
      show p > n
      by_contra ple
      push_neg at ple
      have : p ∣ Nat.factorial (n + 1) := by
        apply Nat.dvd_factorial
        apply pp.pos
        linarith
      have : p ∣ 1 := by
        convert Nat.dvd_sub' pdvd this
        simp
      show False
      have := Nat.le_of_dvd zero_lt_one this
      linarith [pp.two_le]

    </FORMAL>

    Output:
    <STATEMENT>

    For every natural number $n$, there is a prime number $p$ that is larger than $n$.
    Equivalently, there are infinitely many primes.

    </STATEMENT>
    <PROOF>

    First, we fix an arbitrary natural number $n$. 
    Then, we note that $(n + 1)!+1$ is at least $2$, by definition of factorial and addition properties.
    We then note that there exists a prime factor $p$ of $(n + 1)!+1$.
    We aim to show that $p$ is greater than $n$, by first assuming for the sake of contradiction that $p$ is not greater than $n$.
    Then as $p \\le n$, and $p$ is positive, $p$ divides $(n + 1)!$.
    As $p$ divides both $(n + 1)!+1$ and $(n + 1)!$, it must also divide their difference, which is $1$.
    Thus, $p$ must be at most $1$, which contradicts the fact that $p$ is a prime number, as primes are at least $2$.
    Thus, we have a contradiction, and therefore $p$ must be greater than $n$.

    </PROOF>
    </EXAMPLE>

    Now, informalize the following theorem and proof, which is wrapped in <FORMAL>...</FORMAL> tags, and be sure to wrap your informal statement in <STATEMENT>...</STATEMENT> tags and your informal proof in <PROOF>...</PROOF> tags.

    {context + "\n\n" if include_context else ""}<FORMAL>\n{theorem_text}\n</FORMAL>
    """

    return [{"role": "user", "content": prompt}]

def process_response(response):
    """
    Process the response from the model to extract the informal statement and proof.
    The response is expected to be in the format:
    <STATEMENT>...</STATEMENT>
    <PROOF>...</PROOF>
    """
    success = True

    if "<STATEMENT>" in response and "</STATEMENT>" in response:
        statement = response.split("<STATEMENT>")[1].split("</STATEMENT>")[0].strip()
    else:
        success = False
        statement = "No informal statement provided."

    if "<PROOF>" in response and "</PROOF>" in response:
        proof = response.split("<PROOF>")[1].split("</PROOF>")[0].strip()
    else:
        success = False
        proof = "No informal proof provided."

    return statement, proof, success