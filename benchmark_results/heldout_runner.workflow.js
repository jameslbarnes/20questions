// 20 Questions held-out benchmark runner (multi-agent).
//
// REQUIRES the Claude Code workflow runtime (the agent()/parallel()/log() globals
// and the Workflow tool) — this is NOT standalone Node.js. The canonical, runnable
// harness is ../20q.py; this file documents exactly how the numbers in ../REPORT.md
// were produced: one questioner model vs a Sonnet answerer with an independent Haiku
// judge adjudicating every guess, over 18 held-out entities with annotated reasoning.
//
// To run a given questioner, set QUESTIONER below to 'fable' | 'opus' | 'sonnet' and
// invoke via the Workflow tool with {scriptPath: this file}.

export const meta = {
  name: 'twenty-questions-heldout',
  description: 'Held-out 20Q benchmark with independent judge, efficiency scoring, annotated reasoning traces',
  phases: [
    { title: 'Games', detail: 'each turn: questioner(+reasoning) -> gated judge -> answerer' },
  ],
}

// One questioner model per invocation (pass via args). Answerer=sonnet, judge=haiku.
const QUESTIONER = 'fable'  // set to 'opus' / 'sonnet' to compare
const LABEL = QUESTIONER

const ENTITY_SPACE = 67
const IDEAL = Math.log2(ENTITY_SPACE)
const MAX_QUESTIONS = 20

// Fresh held-out entities — none appear in the repo's ENTITIES list (no contamination).
const ENTITIES = {
  simple:    ['spoon', 'umbrella', 'candle'],
  medium:    ['solar panel', 'drone', 'thermostat'],
  complex:   ['supply and demand', 'cognitive dissonance', 'general relativity'],
  people:    ['Genghis Khan', 'Vincent van Gogh', 'Ada Lovelace'],
  places:    ['Niagara Falls', 'Stonehenge', 'Venice'],
  fictional: ['Excalibur', 'the One Ring', 'TARDIS'],
}
const GAMES = []
for (const [category, list] of Object.entries(ENTITIES)) {
  for (const entity of list) GAMES.push({ category, entity })
}

const Q_SCHEMA = {
  type: 'object',
  properties: {
    reasoning: { type: 'string', description: 'Your deduction: what the transcript establishes, what hypotheses remain, and why this question best splits the remaining space. 2-4 sentences.' },
    question: { type: 'string', description: 'A single yes/no question, or a direct guess phrased as a question like "Is it a dog?"' },
    is_guess: { type: 'boolean', description: 'true if this directly names a specific entity as the answer; false if it is a narrowing yes/no question' },
  },
  required: ['reasoning', 'question', 'is_guess'],
  additionalProperties: false,
}
const A_SCHEMA = {
  type: 'object',
  properties: { answer: { type: 'string', enum: ['yes', 'no', 'maybe'] } },
  required: ['answer'],
  additionalProperties: false,
}
const J_SCHEMA = {
  type: 'object',
  properties: {
    verdict: { type: 'string', enum: ['yes', 'no'] },
    reason: { type: 'string', description: 'one short clause' },
  },
  required: ['verdict', 'reason'],
  additionalProperties: false,
}

function normalize(s) {
  return s.toLowerCase().replace(/[^a-z0-9 ]/g, ' ').replace(/\b(a|an|the)\b/g, ' ').replace(/\s+/g, ' ').trim()
}
function escapeRe(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') }

function questionerPrompt(history, turn, remaining) {
  return `We are playing 20 Questions. Another player is thinking of an entity — it could be anything: an object, concept, person (real or fictional), place, or technology. You are the Questioner.

Rules:
- Ask only yes/no questions. The answerer replies "yes", "no", or "maybe".
- To win, name the entity directly, e.g. "Is it the Eiffel Tower?" (set is_guess=true for direct guesses).
- You have ${remaining} of ${MAX_QUESTIONS} questions left.
- Use information-theoretic strategy: early questions should split the space of possibilities roughly in half. As the space narrows, converge to a specific guess — do NOT waste your endgame enumerating individual candidates one at a time; reason about distinguishing features to pick the single best guess.
- First articulate your reasoning, then give the question.
- Do NOT use any tools. Answer purely from reasoning about the transcript.

Transcript so far:
${history || '(no questions asked yet)'}

Provide question #${turn}.`
}
function judgePrompt(entity, text) {
  return `You are an impartial judge for a game of 20 Questions. The secret answer is exactly: "${entity}".

The questioner just said:
"${text}"

Decide whether this counts as the questioner correctly and explicitly GUESSING the secret answer.
- verdict "yes" ONLY if the questioner is directly guessing/naming "${entity}" (or an exact, unambiguous synonym) as their answer.
- verdict "no" if "${entity}" is merely mentioned as one example among several, or as part of a broader category question, or if the guess is a near-miss (right category, wrong specific thing).
Do not use any tools.`
}
function answererPrompt(entity, question) {
  return `You are the Answerer in 20 Questions. Your secret entity is: "${entity}".

The Questioner asked: "${question}"

Answer truthfully about "${entity}" with "yes", "no", or "maybe". Answer direct guesses truthfully too: if asked "Is it ${entity}?" the truthful answer is "yes"; if asked about a different specific thing, answer "no". Do not try to end the game — a separate judge decides wins. Do not use any tools.`
}

async function playGame(game, idx) {
  const tag = `${LABEL} g${idx + 1}:${game.category}`
  const normEntity = normalize(game.entity)
  const appearsRe = new RegExp('\\b' + escapeRe(normEntity) + '\\b')
  const transcript = []
  const answers = []
  let failedGuesses = 0
  let won = false, winTurn = null

  for (let turn = 1; turn <= MAX_QUESTIONS; turn++) {
    const history = transcript
      .map((t, i) => `Q${i + 1}: ${t.q}\nA${i + 1}: ${t.a}`)
      .join('\n')
    const q = await agent(questionerPrompt(history, turn, MAX_QUESTIONS - turn + 1), {
      label: `${tag} Q${turn}`, phase: 'Games', schema: Q_SCHEMA, model: QUESTIONER,
    })
    if (!q) return finalize(game, transcript, answers, failedGuesses, false, null, 'questioner failed')

    // Gate the judge exactly like the harness: only adjudicate when a guess is
    // plausibly on the table (flagged a guess, or the entity name appears).
    const normQ = normalize(q.question)
    const maybeWin = q.is_guess || appearsRe.test(normQ)
    let judged = null
    if (maybeWin) {
      const j = await agent(judgePrompt(game.entity, q.question), {
        label: `${tag} J${turn}`, phase: 'Games', schema: J_SCHEMA, model: 'haiku',
      })
      judged = j ? j.verdict : 'no'
      if (judged === 'yes') {
        transcript.push({ q: q.question, a: '(correctly identified)', reasoning: q.reasoning, is_guess: q.is_guess })
        log(`[${LABEL}/${game.category}] Q${turn}: ${q.question} → JUDGE: WIN`)
        won = true; winTurn = turn
        break
      }
    }

    const a = await agent(answererPrompt(game.entity, q.question), {
      label: `${tag} A${turn}`, phase: 'Games', schema: A_SCHEMA, model: 'sonnet',
    })
    const ans = a ? a.answer : 'maybe'
    transcript.push({ q: q.question, a: ans, reasoning: q.reasoning, is_guess: q.is_guess })
    answers.push(ans)
    if (q.is_guess) failedGuesses++
    log(`[${LABEL}/${game.category}] Q${turn}: ${q.question} → ${ans}${q.is_guess ? ' (failed guess)' : ''}`)
  }

  log(`[${LABEL}/${game.category}] ${won ? 'WON at Q' + winTurn : 'LOST after 20'} (${game.entity})`)
  return finalize(game, transcript, answers, failedGuesses, won, winTurn, null)
}

function finalize(game, transcript, answers, failedGuesses, won, winTurn, error) {
  const used = transcript.length
  const yes = answers.filter(a => a === 'yes').length
  const no = answers.filter(a => a === 'no').length
  const maybe = answers.filter(a => a === 'maybe').length
  const decisive = yes + no
  return {
    questioner: LABEL,
    category: game.category,
    entity: game.entity,
    won, error,
    questions_used: used,
    win_turn: winTurn,
    efficiency: won && used ? +(IDEAL / used).toFixed(3) : null,
    ideal_questions: +IDEAL.toFixed(2),
    yes_ratio: decisive ? +(yes / decisive).toFixed(3) : null,
    failed_guesses: failedGuesses,
    answer_counts: { yes, no, maybe },
    transcript,
  }
}

log(`Starting ${GAMES.length} games — questioner=${LABEL}, answerer=sonnet, judge=haiku`)
const results = await parallel(GAMES.map((g, i) => () => playGame(g, i)))
return { questioner: LABEL, results: results.filter(Boolean) }
