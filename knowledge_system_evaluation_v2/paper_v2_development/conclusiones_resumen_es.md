# Conclusiones principales de la evaluación

## 1. La historia central no debe ser que Graph-RAG gana claramente

La evaluación no demuestra que Graph-RAG supere de forma estadísticamente estable a Vector-only RAG. Los resultados medios muestran pequeñas diferencias a favor de Graph-RAG en algunas métricas, pero los intervalos de confianza por bootstrap cruzan cero en todas las comparaciones principales.

Por tanto, el framing correcto no es:

> Graph-RAG mejora significativamente a Vector-only RAG.

El framing correcto es:

> Graph-RAG es competitivo con Vector-only RAG y muestra pequeñas mejoras observadas en correctness y faithfulness, pero estas mejoras no son estadísticamente estables en la evaluación actual.

Esto es importante porque responde de forma honesta a una de las críticas principales de la primera submission: la necesidad de demostrar claramente el valor añadido del knowledge graph.

## 2. La principal limitación es la cobertura del corpus

La evaluación usa 493 preguntas reales de la comunidad:

- 307 de MOBAFire.
- 186 de Arqade.
- 396 preguntas actuales.
- 97 preguntas legacy.
- 247 preguntas Full, 124 Partial y 122 None según answerability.

El resultado más claro es que el corpus de vídeos procesados no cubre suficientemente la variedad de preguntas reales que hacen los jugadores. Por eso todos los sistemas basados en retrieval rechazan responder muchas veces:

- Graph-RAG: 82.35% de refusal.
- Vector-only RAG: 82.96%.
- ASR-only: 83.16%.
- Vision-only: 84.38%.
- BM25: 86.21%.

La conclusión no debe ser que el sistema simplemente funciona mal, sino que la base de conocimiento es todavía pequeña frente a la amplitud del benchmark.

Framing recomendado:

> La evaluación muestra que la cobertura del corpus es el principal cuello de botella. Muchas preguntas reales no pueden responderse de forma grounded porque la información necesaria no está presente en los vídeos procesados.

## 3. El sistema es conservador y evita responder sin evidencia

Aunque la tasa de refusal es alta, la tasa de over-refusal es muy baja. En Graph-RAG, Vector-only, ASR-only y BM25 es aproximadamente 0.41%.

Esto significa que, cuando el sistema se niega a responder, normalmente el evaluador también considera que el contexto recuperado no contiene evidencia suficiente.

Esta es una conclusión positiva:

> Los sistemas basados en retrieval muestran abstención calibrada: prefieren no responder cuando no hay evidencia suficiente en lugar de inventar una respuesta.

Esto ayuda a diferenciar el sistema de un modelo puramente paramétrico que responde casi siempre, pero sin grounding en los vídeos.

## 4. Los modelos paramétricos responden mejor, pero no son grounded

Los baselines sin contexto, especialmente `sota_base`, tienen mucha más correctness y relevance:

- `sota_base`: correctness 1.199 sobre 2, relevance 0.839.
- `vanilla_base`: correctness 0.740, relevance 0.762.
- Graph-RAG: correctness 0.144, relevance 0.250.
- Vector-only: correctness 0.134, relevance 0.266.

Esto no significa que el sistema RAG sea inútil. Significa que los modelos paramétricos tienen mucho conocimiento general sobre League of Legends y pueden contestar muchas preguntas desde memoria interna, sin depender de los vídeos.

El paper debe separar claramente dos cosas:

- Conocimiento general del juego.
- QA grounded sobre un corpus concreto de vídeos procesados.

Framing recomendado:

> Los modelos paramétricos funcionan como una referencia de conocimiento general, pero no cumplen el objetivo principal del sistema: responder únicamente a partir de evidencia recuperada del corpus audiovisual.

## 5. Graph-RAG es competitivo con Vector-only, pero no superior de forma concluyente

Comparación global Graph-RAG frente a Vector-only:

| Métrica | Graph-RAG | Vector-only | Diferencia | 95% CI |
|---|---:|---:|---:|---:|
| BERTScore-F1 | 0.734 | 0.736 | -0.002 | [-0.003, +0.000] |
| Correctness | 0.144 | 0.134 | +0.010 | [-0.022, +0.043] |
| Faithfulness | 0.632 | 0.626 | +0.006 | [-0.026, +0.037] |
| Relevance | 0.250 | 0.266 | -0.016 | [-0.055, +0.024] |
| Refusal rate | 82.35% | 82.96% | -0.61 pp | [-4.06 pp, +2.84 pp] |

Todas las diferencias tienen intervalos que cruzan cero. Por tanto:

- No podemos decir que Graph-RAG sea significativamente mejor.
- Sí podemos decir que es competitivo.
- Sí podemos decir que muestra pequeñas mejoras observadas en correctness y faithfulness.
- Sí debemos decir que estas mejoras no son concluyentes.

Framing recomendado:

> Graph-RAG se comporta de forma competitiva respecto a Vector-only RAG, pero la evaluación actual no proporciona evidencia estadísticamente estable de una mejora global.

## 6. El patrón más prometedor aparece en preguntas Partial

El resultado más interesante para defender el knowledge graph aparece en las preguntas `Partial`, donde existe algo de evidencia relevante, pero no una respuesta directa y completa.

En ese subconjunto:

| Métrica | Graph-RAG | Vector-only | Diferencia | 95% CI |
|---|---:|---:|---:|---:|
| Correctness | 0.185 | 0.145 | +0.040 | [-0.032, +0.121] |
| Faithfulness | 0.648 | 0.622 | +0.026 | [-0.033, +0.086] |
| Relevance | 0.258 | 0.251 | +0.007 | [-0.079, +0.096] |
| Refusal rate | 83.87% | 83.06% | +0.81 pp | [-5.65 pp, +7.26 pp] |

Aquí Graph-RAG tiene las mejores diferencias observadas. Esto sugiere que el grafo puede ayudar cuando la información está fragmentada entre entidades, relaciones y chunks distintos.

Pero los intervalos de confianza también cruzan cero. Por tanto, hay que presentarlo como un indicio, no como una prueba.

Framing recomendado:

> El mayor beneficio observado de Graph-RAG aparece en preguntas parcialmente answerable, lo que sugiere que el grafo puede ser útil cuando la evidencia está fragmentada. Sin embargo, esta mejora no es estadísticamente estable en el conjunto actual.

## 7. BM25 es un baseline fuerte para entidades exactas

BM25 obtiene la mayor faithfulness media entre los sistemas de retrieval y también el mayor recall exacto de entidades en `retrieval_metrics.json`.

Esto tiene sentido: si la respuesta gold contiene nombres exactos de campeones, objetos, runas o hechizos, BM25 puede encontrarlos bien por coincidencia léxica.

Pero BM25 también tiene:

- La mayor refusal rate.
- Baja correctness.
- Menor capacidad para razonamiento semántico o relacional.

Framing recomendado:

> BM25 sigue siendo un baseline fuerte para recuperación léxica exacta en dominios con muchas entidades nombradas, pero no basta para QA grounded más semántico o relacional.

## 8. Audio y visión aportan señales complementarias

Las ablations muestran que ni ASR-only ni Vision-only son suficientes por separado.

El análisis sugiere:

- La visión ayuda más a identificar el sujeto o campeón principal.
- El audio contiene más vocabulario estratégico: objetos, builds, runas, habilidades y explicaciones.

Esto justifica mantener una arquitectura multimodal.

Framing recomendado:

> La información visual y la información hablada cumplen funciones distintas: la visión ancla el contexto de gameplay, mientras que el audio aporta muchos de los detalles estratégicos necesarios para responder.

## 9. Correctness debe interpretarse junto con refusal

Los sistemas de retrieval tienen muchas respuestas con correctness 0. Pero muchas de esas respuestas son refusals, no alucinaciones.

Por ejemplo:

- Graph-RAG tiene 429 respuestas con score 0, 57 con score 1 y 7 con score 2.
- Vector-only tiene 435 con score 0, 50 con score 1 y 8 con score 2.
- `sota_base` tiene muchos más scores 1 y 2 porque responde siempre desde conocimiento paramétrico.

Por eso no se debe interpretar correctness sola. En este tipo de sistema, una refusal puede ser el comportamiento correcto si no hay evidencia.

Framing recomendado:

> En QA grounded con un corpus incompleto, la correctness mezcla dos fenómenos: respuestas incorrectas y abstenciones justificadas. Por eso hay que analizarla junto con refusal y over-refusal.

## 10. Qué debemos afirmar en el paper

Sí podemos afirmar:

- La nueva evaluación es mucho más fuerte que la de la primera submission.
- Usa preguntas reales de la comunidad.
- Separa preguntas Current y Legacy.
- Separa preguntas Full, Partial y None según answerability.
- Incluye baselines paramétricos, BM25, ASR-only, Vision-only, Vector-only y Graph-RAG.
- Muestra que la cobertura del corpus es el cuello de botella principal.
- Muestra que los sistemas de retrieval son conservadores y evitan responder sin evidencia suficiente.
- Muestra que Graph-RAG es competitivo con Vector-only.
- Muestra un patrón prometedor para Graph-RAG en preguntas Partial.
- Justifica la necesidad de combinar audio y visión.

No debemos afirmar:

- Que Graph-RAG mejora significativamente a Vector-only.
- Que el knowledge graph queda demostrado como claramente superior.
- Que el sistema resuelve completamente el problema de QA sobre vídeos de gameplay.
- Que las métricas de correctness bajas significan únicamente respuestas incorrectas.

## 11. Framing final recomendado

El mensaje final debería ser algo como:

> La evaluación con preguntas reales muestra que el principal reto para QA grounded sobre vídeos de gameplay es la cobertura del corpus. Los modelos paramétricos responden más preguntas correctamente, pero sin grounding en los vídeos. En cambio, los sistemas de retrieval son mucho más conservadores y se abstienen cuando no encuentran evidencia suficiente. Graph-RAG es competitivo con Vector-only RAG y muestra pequeñas mejoras observadas en correctness y faithfulness, especialmente en preguntas parcialmente answerable, pero los intervalos de confianza cruzan cero y no permiten afirmar una mejora estadísticamente estable. Estos resultados apoyan la viabilidad del enfoque, pero también muestran que la escala del corpus, la cobertura de retrieval y una evaluación más dependiente de relaciones de grafo son los próximos puntos clave.

