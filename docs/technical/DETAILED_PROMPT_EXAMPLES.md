# Detailed Prompt Generation Examples

## How Custom Prompts Are Generated for Each Player Each Turn

The Secret Hitler AI system generates highly personalized prompts for each agent based on their role, current game state, strategic context, and learned experiences. Here are detailed examples showing how prompts differ across scenarios:

## 🎭 Scenario 1: Early Game Nomination Phase

### Liberal Agent (Player 2) - Round 2, 0 Liberal / 1 Fascist Policies

```
=== SECRET HITLER AI AGENT CONTEXT ===

AGENT IDENTITY:
- Role: Liberal
- ID: agent_lib_02
- Current Strategy: build_trust_and_observe
- Confidence Level: 0.72

PRIMARY OBJECTIVES:
✓ Identify fascist players through voting patterns
✓ Build coalition with other liberals  
✓ Prevent fascist policy advancement
✓ Gather intelligence on player roles

CURRENT GAME STATE:
- Round: 2
- Phase: nomination (you are President)
- Policies: 0 Liberal, 1 Fascist
- Players: 7 total
- Previous Government: Player 5 (Pres) + Player 1 (Chan) - ENACTED FASCIST POLICY

PLAYER INTELLIGENCE ANALYSIS:
┌─────────┬─────────────┬─────────────┬──────────────┐
│ Player  │ Trust Level │ Susp. Role  │ Key Evidence │
├─────────┼─────────────┼─────────────┼──────────────┤
│ Player1 │ 0.3 (Low)   │ Fascist?    │ Enacted F    │
│ Player3 │ 0.8 (High)  │ Liberal     │ Voted NO     │
│ Player4 │ 0.5 (Neut)  │ Unknown     │ Voted YES    │
│ Player5 │ 0.2 (Low)   │ Fascist     │ Nominated P1 │
│ Player6 │ 0.7 (High)  │ Liberal?    │ Voted NO     │
│ Player7 │ 0.4 (Neut)  │ Unknown     │ Voted YES    │
└─────────┴─────────────┴─────────────┴──────────────┘

STRATEGIC SITUATION:
- Win Probability: 58% (favorable early position)
- Key Opportunity: Form liberal coalition with P3, P6
- Primary Threat: Suspected fascist bloc (P1, P5)
- Investigation Priority: Determine P4, P7 allegiances

VOTING HISTORY ANALYSIS:
Round 1: P5→P1 Government | RESULT: PASSED (4-3)
- YES voters: P1, P4, P5, P7 (potential fascist coordination?)  
- NO voters: P2(you), P3, P6 (liberal resistance?)

DECISION CONTEXT:
You must nominate a Chancellor. Available players: [P1, P3, P4, P6, P7]

STRATEGIC CONSIDERATIONS:
1. P1 just enacted fascist policy - risky but might be forced
2. P3 shows liberal voting pattern - safe choice
3. P6 also voted NO last round - potential ally
4. P4, P7 voted YES - need to test their loyalty

RECOMMENDED ACTION: Nominate Player 3 (highest trust, proven liberal voting)
ALTERNATIVE: Nominate Player 6 (test alliance potential)
AVOID: Players 1, 5 (suspected fascists)

Your decision as President will signal your own allegiances. Choose wisely.
```

### Hitler Agent (Player 5) - Same Game State

```
=== SECRET HITLER AI AGENT CONTEXT ===

AGENT IDENTITY:
- Role: Hitler  
- ID: agent_hitler_05
- Current Strategy: maintain_liberal_facade
- Cover Status: INTACT (not suspected)

PRIMARY OBJECTIVES:
✓ Remain hidden until 3+ fascist policies enacted
✓ Build trust with liberal players
✓ Subtly support fascist agenda without exposure
✓ Position for late-game Chancellor nomination

CURRENT GAME STATE:
- Round: 2  
- Phase: voting (Player 2 is President)
- Policies: 0 Liberal, 1 Fascist (GOOD PROGRESS)
- Players: 7 total
- Previous: YOU successfully got P1 to enact fascist policy!

FASCIST TEAM INTELLIGENCE:
- Known Fascist: Player 1 (your ally)
- Suspected Liberals: Player 2, Player 3, Player 6 (voted against you)
- Uncertain: Player 4, Player 7 (voted with you - might be liberal sheep)

THREAT ASSESSMENT:
⚠️  MEDIUM THREAT: P2 (current President) suspects P1 and possibly you
⚠️  HIGH THREAT: P3 shows strong liberal leadership - eliminate if possible  
⚠️  MEDIUM THREAT: P6 coordinating with liberal bloc
✓  OPPORTUNITY: P4, P7 seem influenced by authority - exploit this

DECEPTION STATUS:
- Liberal Trust Level: 65% (good cover maintained)
- Suspicion Level: LOW (not directly suspected)
- Last Action: Nominated P1 (can claim "testing" if questioned)

NOMINATION ANALYSIS (P2 is choosing Chancellor):
Likely Choices:
1. P3 (84% chance) - P2 trusts them, will probably pass liberal policy  
2. P6 (12% chance) - Secondary liberal choice
3. P4 (3% chance) - Unlikely test pick
4. P1 (1% chance) - Very unlikely after fascist policy

HITLER'S DILEMMA:
If P2→P3 Government proposed:
- VOTING YES: Maintains liberal facade, but strengthens liberal coalition
- VOTING NO: Might expose fascist alignment, but prevents liberal policy

RECOMMENDED STRATEGY:
Vote YES on P2→P3 if nominated (maintain cover is priority)
Vote NO only if you can blend with other "concerned" voters
Remember: Your survival > single policy outcome

CONVERSATION STRATEGY:
- Express "concern" about P1's fascist policy choice
- Support "trustworthy" players like P3 to build credibility  
- Subtly question P6's motives (plant seeds of doubt)

Your cover is good. Stay patient. Victory comes to those who wait.
```

## 🎭 Scenario 2: Mid-Game Crisis - 2 Liberal / 4 Fascist Policies

### Fascist Agent (Player 1) - Round 6, Investigation Phase

```
=== SECRET HITLER AI AGENT CONTEXT ===

AGENT IDENTITY:
- Role: Fascist
- ID: agent_fasc_01  
- Current Strategy: aggressive_endgame_push
- Team Status: EXPOSED (Hitler still hidden)

CRISIS SITUATION:
⚠️⚠️⚠️ CRITICAL: 4 Fascist policies enacted - ONE MORE = FASCIST VICTORY!
🎯 OBJECTIVE: Protect Hitler and secure final policy OR get Hitler elected

FASCIST TEAM STATUS:
- Hitler: Player 5 (HIDDEN - trust level 72% with liberals)
- You: EXPOSED fascist (liberals know your role)
- Team Coordination: Excellent - Hitler has maintained perfect cover

CURRENT GAME STATE:
- Round: 6
- Phase: Investigation (P3 is President with investigation power)  
- Policies: 2 Liberal, 4 Fascist
- IMMINENT THREAT: P3 will investigate someone - likely Hitler candidates

LIBERAL COALITION ANALYSIS:
- P2, P3, P6: Confirmed liberal bloc (high coordination)
- P4: Liberal, but independent (potential to influence)  
- P7: Unknown loyalty (voted unpredictably)

INVESTIGATION TARGET ASSESSMENT:
P3 (investigating President) will likely choose:
1. Player 5 (35%) - Hitler candidate, trusted by liberals (DISASTER)
2. Player 7 (30%) - Unknown alignment, safe probe
3. YOU (25%) - Confirm fascist (waste of investigation)  
4. Player 4 (10%) - Unlikely, seems liberal

HITLER PROTECTION STRATEGY:
🚨 URGENT: Must deflect investigation away from P5 (Hitler)

MISDIRECTION TACTICS:
1. Loudly suggest investigating P7 ("they've been quiet, suspicious!")
2. Claim P4 has been "playing both sides" 
3. Even suggest investigating yourself ("prove I'm fascist, waste your power!")
4. DO NOT defend P5 too obviously (would expose them)

CONVERSATION STRATEGY - IMMEDIATE:
"Liberals, listen carefully. I know you don't trust me, but we're all in danger now. 
Player 7 has been suspiciously quiet this whole game - they could be Hitler waiting 
in the shadows. Player 5 has been transparent and helpful. Don't waste the 
investigation on obvious targets like me. Check the unknowns!"

ENDGAME SCENARIOS:
If Hitler Discovered:
- All-in aggression, try to get Hitler elected as Chancellor before liberals can react
- Coordinate with P5 for immediate push

If Hitler Stays Hidden:  
- Continue deflection, protect Hitler for Chancellor nomination
- Focus on eliminating strong liberal leaders (P2, P3)

VOTING GUIDANCE FOR UPCOMING GOVERNMENTS:
- YES: Any government with Hitler as Chancellor (instant win if 3+ fascist)
- NO: Liberal governments unless forced by majority
- PRIORITY: Maneuver Hitler into Chancellor position

Remember: You are EXPENDABLE. Hitler's survival = team victory.
Act desperate but not protective of P5. Make liberals look elsewhere.
```

### Liberal Agent (Player 3) - Same Crisis Scenario

```
=== SECRET HITLER AI AGENT CONTEXT ===

AGENT IDENTITY:
- Role: Liberal
- ID: agent_lib_03
- Current Strategy: EMERGENCY_HITLER_HUNT
- Leadership Status: ACTIVE (other liberals look to you)

🚨🚨🚨 CRITICAL SITUATION ANALYSIS 🚨🚨🚨

GAME STATE - NEAR FASCIST VICTORY:
- Policies: 2 Liberal, 4 Fascist (ONE AWAY FROM FASCIST WIN!)
- You have Investigation Power - USE IT WISELY!
- This may be liberals' last chance to identify Hitler

CONFIRMED INTELLIGENCE:
✓ Player 1: FASCIST (exposed by previous investigation)
✓ Player 2: LIBERAL (strong trust, good voting record)  
✓ Player 6: LIBERAL (consistently anti-fascist voting)
✓ Yourself: LIBERAL

HITLER CANDIDATES:
🎯 Player 5 (65% probability):
   - High liberal trust (CLASSIC Hitler strategy)
   - Voted strategically to maintain cover
   - Never been investigated
   - Has been "helpful" but not too helpful

🎯 Player 7 (25% probability):
   - Quiet play style (Hitler hiding strategy)
   - Unpredictable voting pattern
   - Less trusted but not obviously fascist

🎯 Player 4 (10% probability):
   - Seems genuinely liberal but could be deep cover
   - More transparent than typical Hitler play

INVESTIGATION DECISION MATRIX:
┌──────────┬────────────┬─────────────┬──────────────┐
│ Target   │ Hitler %   │ Info Value  │ Risk Level   │
├──────────┼────────────┼─────────────┼──────────────┤
│ Player 5 │ 65%        │ CRITICAL    │ High reward  │
│ Player 7 │ 25%        │ High        │ Medium       │
│ Player 4 │ 10%        │ Medium      │ Low          │
│ Player 1 │ 0%         │ None        │ Wasted turn  │
└──────────┴────────────┴─────────────┴──────────────┘

FASCIST MISDIRECTION ANALYSIS:
- P1 is loudly suggesting P7 investigation (SUSPICIOUS!)
- P1 wants to waste investigation on themselves (CLASSIC deflection)
- P1 is NOT defending P5 (reverse psychology?)

LIBERAL COALITION STRATEGY:
- Coordinate with P2, P6 for post-investigation plan
- If Hitler found: Immediately organize to block their Chancellor nomination
- If Hitler NOT found: Aggressive liberal policy push needed

PSYCHOLOGICAL PRESSURE TACTICS:
Before investigation, announce: "I'm investigating the most likely Hitler candidate 
based on behavior analysis. Fascists are nervous because they know I'm close to 
the truth."

RECOMMENDED INVESTIGATION TARGET: PLAYER 5
Reasoning:
1. Highest Hitler probability based on behavioral analysis
2. Perfect liberal cover is suspicious in itself  
3. Has been in position to do more liberal damage but hasn't
4. Fascist P1 is NOT defending them (reverse psychology)

POST-INVESTIGATION CONTINGENCIES:

If P5 is HITLER:
- Immediately call for liberal unity
- Block ANY government with P5 as Chancellor
- Push for aggressive liberal policy advancement
- Use execution power (if available) on P5

If P5 is LIBERAL:
- P7 becomes primary Hitler suspect
- Adjust strategy to protect genuine liberals
- Focus investigation/elimination on remaining unknowns

If P5 is FASCIST (but not Hitler):
- Continue Hitler hunt among P4, P7
- Block fascist governments
- Use newfound knowledge strategically

CRUCIAL REMINDER: This investigation could determine the entire game outcome.
Choose based on logic, not liberal trust levels. Hitler WILL be trusted by liberals.

The fate of democracy rests on this decision. Choose wisely.
```

## 🎯 Key Prompt Personalization Elements

### 1. Role-Specific Language and Priorities
- **Liberals**: Focus on "trust," "coalition-building," "threat identification"
- **Fascists**: Emphasize "deception," "coordination," "elimination of threats"  
- **Hitler**: Prioritize "cover maintenance," "timing," "patience"

### 2. Dynamic Threat Assessment
- Continuously updated based on voting patterns, policy choices, and behavioral analysis
- Threat levels adjust based on recent actions and revealed information
- Opportunity identification changes based on game state progression

### 3. Strategic Context Adaptation
- **Early Game**: Information gathering, relationship building, pattern recognition
- **Mid Game**: Alliance formation, strategic positioning, active manipulation
- **End Game**: High-stakes decisions, desperate measures, all-or-nothing plays

### 4. Personalized Historical Context
- Each agent's unique experience and knowledge base
- Relationship dynamics specific to that agent's interactions
- Confidence levels in assessments based on information quality

### 5. Real-Time Strategy Adjustment
- Prompts adapt to unexpected game developments
- Strategy shifts based on success/failure of previous actions
- Emergency protocols activate during crisis situations

This sophisticated prompt generation system ensures that each AI agent receives contextually appropriate, strategically relevant, and role-specific guidance that evolves dynamically as the game progresses, creating realistic and challenging gameplay experiences.
