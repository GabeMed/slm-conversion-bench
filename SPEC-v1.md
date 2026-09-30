# SPEC — slm-conversion-bench

**Execução empírica de "Small Language Models are the Future of Agentic AI"**
(Belcak et al., NVIDIA Research, arXiv 2506.02153)

## Como ler este documento

O paper é um _position paper_: afirma que SLMs são o futuro da IA agêntica, sustenta a posição com argumentos e propõe um algoritmo de conversão LLM→SLM, mas não contém nenhum experimento. Os "estudos de caso" do apêndice são estimativas.

Esta prova de conceito executa o algoritmo do paper de ponta a ponta numa tarefa agêntica real e transforma cada argumento em uma quantidade medida, com um único objetivo: **descobrir se melhora, quanto melhora e qual passo do algoritmo entrega o ganho**.

Cada seção deste documento espelha uma seção do paper. Os rótulos **V** (visões), **A** (argumentos), **AV** (visões alternativas), **CA** (contra-argumentos), **B** (barreiras), **S** (passos do algoritmo) e **WD** (definições) são os do paper. Os rótulos **D** são decisões nossas, listadas na Seção 9.

Regra de nível: esta spec não nomeia modelos, provedores nem bibliotecas. Esses vão em arquivos de configuração e em decisões de implementação, tomadas nas sessões de desenvolvimento. Aqui, "o LLM de produção", "o professor" e "os SLMs" são papéis.

Para cada item do paper, o documento responde três perguntas: **o que o paper diz**, **o que executamos** e **o que isso reproduz**. Quando um item não é testável nesta PoC, isso é dito explicitamente.

---

## 1. Posição (Seção 2 do paper)

### Definições

**Paper.** WD1: um SLM é um modelo que cabe num dispositivo eletrônico de consumo e faz inferência com latência prática para servir as requisições agênticas de um usuário. WD2: um LLM é um modelo que não é SLM. Em 2025, o paper considera SLM a maioria dos modelos abaixo de 10 bilhões de parâmetros.

**Executamos.** Adotamos WD1 e WD2. Os SLMs candidatos são organizados em três faixas de tamanho (até 1B, de 1B a 4B, de 7B a 9B), e a pergunta implícita em toda medição é "até onde dá para descer". Uma rodada de inferência em máquina de consumo é opcional, para honrar WD1 literalmente (D17).

### As três visões

O paper afirma que os SLMs são (V1) suficientemente poderosos, (V2) inerentemente mais adequados e (V3) necessariamente mais econômicos para a maioria das invocações em sistemas agênticos. Ele apresenta a posição como uma consequência necessária dessas três visões, e chega a chamá-la de um "ought" moral. Esta PoC não testa o valor; testa as premissas empíricas.

| Visão                        | O que o paper afirma                                     | A medida que a fecha                                                                                                            |
| ---------------------------- | -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| **V1** Suficiência           | SLMs dão conta dos "errands" de linguagem dos agentes    | Qualidade do SLM contra o LLM de produção no benchmark, zero-shot e após especialização, por categoria de dificuldade           |
| **V2** Adequação operacional | SLMs são mais adequados a sistemas agênticos do que LLMs | Não é medida direta: é a soma de A3 (tempo de adaptação), A5 (aderência a formato) e A6 (sistema misto)                         |
| **V3** Economia              | SLMs são mais econômicos pela virtude do tamanho         | Custo por mil chamadas com throughput medido, mais utilização e custo fixo (os pontos que o paper admite deixar de fora em AV2) |

---

## 2. Argumentos (Seção 3 do paper)

### A1 — SLMs já são suficientemente poderosos

**Paper.** Cita modelos de 1 a 9 bilhões de parâmetros que igualam modelos muito maiores em raciocínio de senso comum, tool calling, geração de código e instruction following. Toda a evidência vem de benchmarks de terceiros; nenhuma vem de dentro de um pipeline agêntico. O paper nomeia os benchmarks que considera relevantes para a interface modelo→ferramenta e código→modelo.

**Executamos.** A tarefa da PoC é a que o próprio A1 coloca no centro: **tool calling**, medida no benchmark de function calling que o paper cita (BFCL). Como as categorias de turno único desse benchmark estão perto da saturação, a diferença entre SLM e LLM é medida nas categorias vivas e de múltiplos turnos, onde ela ainda aparece (D9). O primeiro dado é o zero-shot de candidatos das três faixas de tamanho, sob o mesmo prompt de produção usado pelo LLM.

**Reproduz.** Verifica se as afirmações do A1 sobrevivem no benchmark que o paper escolheu, no nível de dificuldade em que ainda há o que medir.

**Fora.** Raciocínio de senso comum e geração de código não voltada a ferramentas: o paper os usa como indicadores, não como tarefas de agente.

### A2 — SLMs são mais econômicos

**Paper.** Quatro sub-argumentos: (i) inferência 10–30× mais barata que a de um LLM, citando terceiros; (ii) fine-tune em horas de GPU; (iii) implantação em dispositivos de borda; (iv) uso mais eficiente dos parâmetros, porque LLMs ativam só uma fração deles por entrada.

**Executamos.** Medimos (i) e (ii): custo de inferência a partir do throughput medido numa GPU de referência, e tempo e custo de cada especialização. O item (iii) é opcional (D17).

**Reproduz.** Substitui o "10–30×" citado por um número da tarefa, e diz explicitamente que parte do A2 fica sem teste.

**Fora.** O item (iv). Exige instrumentação interna dos modelos e não é consequência do pipeline.

### A3 — SLMs são mais flexíveis

**Paper.** Por serem baratos de treinar e adaptar, permitem múltiplos especialistas por rotina e adaptação rápida a novas exigências, "overnight rather than weeks". Cita a democratização como consequência.

**Executamos.** Registramos o tempo e o custo de cada especialista treinado, e o esforço necessário para trocar de tarefa (o que muda quando só a especificação da tarefa muda).

**Reproduz.** Mede o "overnight".

**Fora.** A democratização não é testável numa PoC.

### A4 — Agentes expõem só uma fatia estreita do modelo

**Paper.** Um agente é um LLM generalista amarrado por prompts e gerenciamento de contexto a uma função estreita; um SLM ajustado para esses prompts bastaria. O paper remete a objeção natural (o generalista entende melhor o mundo) para a AV1.

**Executamos.** É a premissa do workload: um prompt de produção realista, fixo e versionado antes de qualquer resultado, o mesmo para o LLM e para os SLMs. O S3 mede a estreiteza, contando quantos tipos distintos de chamada existem de fato nos logs.

**Reproduz.** O A4 é verdadeiro se uma especialização estreita fecha o gap. Essa é exatamente a comparação SLM especializado contra LLM sob o mesmo prompt.

### A5 — Interações agênticas exigem alinhamento de formato

**Paper.** O agente precisa de um único formato de tool call e de saída; o LLM que alterna formatos comete erros ocasionais de formato; um SLM treinado num único formato é preferível.

**Executamos.** Taxa de chamadas inválidas contra o schema da ferramenta em todos os braços: LLM de produção, SLM zero-shot, SLM especializado; cada um com e sem decoding restrito ao schema.

**Reproduz.** Separa o ganho de formato que vem do treino do que vem de truque de decoding. É o único ponto em que o paper aposta que o SLM supera o LLM.

### A6 — Sistemas agênticos são naturalmente heterogêneos

**Paper.** Um modelo pode ser ferramenta de outro; a arquitetura natural tem um LLM na raiz (conversa e orquestração) e SLMs nas chamadas subordinadas; modelos de tamanhos diferentes para complexidades diferentes.

**Executamos.** É onde os papéis da PoC são definidos:

- **O LLM de produção**: o modelo que o cliente usa hoje. É o teto de qualidade e a referência de custo. Só avalia; nenhuma saída sua vira alvo de treino.
- **O professor**: o modelo cujas chamadas são registradas e viram dados de especialização. Precisa de licença que permita esse uso.
- **Os SLMs**: os candidatos das três faixas.

O sistema misto do A6 é o SLM atendendo por padrão e o LLM de produção recebendo as chamadas em que o SLM falha. Varremos a taxa de fallback de 0% a 100% e traçamos a fronteira de Pareto qualidade×custo.

**Reproduz.** Testa a recomendação arquitetural central do paper: o sistema misto domina os dois sistemas puros?

**Nota.** A diferença de qualidade entre o professor e o LLM de produção é reportada como **custo de conformidade**: o preço, em qualidade, de usar como fonte de dados um modelo cuja licença permite treinar.

### A7 — Interações agênticas são fonte de dados

**Paper.** Um "listener" na interface de chamada gera dados de instrução especializados; os dados podem ser pós-filtrados pelo sucesso geral do workflow; produzir SLMs especialistas é um passo natural da implantação, não um esforço auxiliar.

**Executamos.** Os logs do S1, a ablação "logs reais contra dados sintéticos" (S1) e o filtro do S2.

**Reproduz.** É a pergunta do fosso de dados: se dados sintéticos gerados só a partir da descrição da tarefa empatam com logs reais, o cliente não precisa entregar logs.

---

## 3. Visões alternativas (Seção 4 do paper)

### AV1 — O generalista sempre vence na mesma tarefa

**Paper.** Sustentada por CA1 (as leis de escala: modelos maiores vencem em qualquer tarefa de linguagem) e CA2 (o "hub semântico": LLMs abstraem e integram significado de um modo que modelos pequenos não conseguem). O paper responde com A8 (as leis de escala assumem arquitetura constante; SLMs se beneficiam de arquiteturas próprias), A9 (a flexibilidade do SLM permite ajustá-lo até a confiabilidade desejada), A10 (compute em inferência é barato para SLMs) e A11 (agentes decompõem problemas em sub-tarefas simples demais para o hub importar).

**Executamos.**

- **A9** é a curva de aprendizado do S5: a especialização chega ao nível do LLM de produção, e com quantos exemplos?
- **A11** é o gap por categoria de dificuldade do benchmark. Se a diferença entre SLM e LLM se concentra nas categorias de múltiplos turnos, o argumento do hub tem força; se é uniforme, não tem.
- **A10** é um braço opcional (D15): o SLM com raciocínio ou auto-consistência em inferência, medido também em latência e custo, porque é aí que o A10 cobra seu preço.

**Reproduz.** O paper rebate a AV1 com argumentos; nós a rebatemos, ou confirmamos, com o gap medido.

**Fora.** **A8** exige comparar SLMs de arquiteturas diferentes em tamanho parecido; é um experimento à parte. Um ponto de dado barato é possível se a lista de candidatos incluir um modelo de arquitetura híbrida (D16).

### AV2 — A centralização torna o LLM mais barato

**Paper.** Sustentada por CA3 (é difícil manter um endpoint especialista bem utilizado e balanceado) e CA4 (custo de infraestrutura e de talento para operá-la, omitido dos cálculos). O paper reconhece a AV2 como válida e diz que "o júri ainda está fora", apontando A12 (avanços em escalonamento de inferência) e A13 (custo de infra em queda).

**Executamos.** O modelo de custo da Seção 7 responde CA3 e CA4 diretamente: custo recalculado a 20%, 50% e 100% de utilização; custo fixo de engenharia e de re-treino incluído no payback; payback em três volumes de referência.

**Reproduz.** É o ponto mais fraco do paper, por confissão dele. A PoC produz os primeiros números ali.

**Fora.** A12 e A13 são tendências de mercado, não testáveis aqui. Nosso custo usa infraestrutura alugada, o que equivale a assumir custo de setup zero; isso fica registrado.

### AV3 — Mundos igualmente possíveis

**Paper.** Os dois mundos são possíveis, mas a inércia e o investimento já feito favorecem o mundo LLM. O paper reconhece a possibilidade.

**Executamos.** Não é testável. O registro de esforço (Seção 7) é o único dado produzido sobre o custo de trocar de mundo.

---

## 4. Barreiras (Seção 5 do paper)

**B1 — Investimento em infraestrutura centralizada.** Não testável.

**B2 — SLMs são projetados e avaliados com benchmarks generalistas.** O paper diz que, olhando só a utilidade agêntica, SLMs superam modelos maiores. Executamos: a ablação do S4, seleção por scores públicos de benchmark contra seleção pelos próprios logs. Reproduz: testa se o ranking generalista prevê o desempenho na tarefa.

**B3 — Falta de atenção pública.** Não testável. A publicação dos resultados é a resposta que o paper pede.

---

## 5. Algoritmo de conversão (Seção 6 do paper)

O algoritmo do paper pressupõe duas coisas que ele não diz: que existe um agente em produção gerando chamadas, e que existe um jeito de avaliar e decidir. O paper não define avaliação, critério de aceite nem o "router" que aparece pela primeira vez no S6. A Seção 7 deste documento cobre o que falta.

**O agente de produção.** É um workload: um conjunto de chamadas de tool calling, com prompt de produção fixo, que o professor responde. Os inputs vêm de um dataset aberto de function calling, distinto do benchmark, cujos rótulos não são usados (D13). O benchmark fica intocado como conjunto de teste, para que os números do SLM e do LLM sejam comparáveis aos publicados.

**Desenho da ablação.** O pipeline completo é o braço principal. Para cada passo há um braço que o substitui pelo que um time faria sem o paper, mantendo tudo o mais igual; a diferença entre os dois é a contribuição do passo. A forma é "tira um de cada vez", e não cumulativa, porque os passos interagem (o S5 depende do S1). Cada braço reporta a diferença de qualidade, a diferença de custo e o esforço para executar o passo.

### S1 — Coleta de logs

**Paper.** Instrumentar todas as chamadas não-HCI do agente, capturando prompt de entrada, resposta, conteúdo das tool calls e, opcionalmente, latência; pipeline criptografado, com controle de acesso e anonimização.

**Executamos.** Um logger em volta das chamadas do professor sobre os inputs de treino do workload. Cada registro guarda entrada estruturada, saída, latência, tokens e a identificação e versão do modelo, num esquema neutro entre provedores (D7). O raciocínio do professor, quando exposto, fica fora do log (D5): é o que o paper descreve e é o que um cliente com modelo fechado teria.

**Ablação.** Logs reais contra dados sintéticos gerados só a partir da descrição da tarefa.

**Reproduz.** A7.

### S2 — Curadoria e filtragem

**Paper.** Acumular 10k–100k exemplos ("regra de bolso" para fine-tune de modelos pequenos); remover PII, PHI e dados sensíveis da aplicação; parafrasear entradas específicas para ofuscar entidades.

**Executamos.** O mascaramento de dados sensíveis é executado como passo real, mesmo com efeito quase nulo em dados de benchmark; o efeito é registrado. Filtro por chamada válida contra o schema (o único sinal de "sucesso" disponível em produção; usar o gabarito seria trapaça). Deduplicação. Split de treino e validação por cenário. A curva de dados do S5 (500 / 2k / 5k / tudo) testa o piso de 10k.

**Ablação.** Logs crus contra logs curados.

**Reproduz.** Diz se a curadoria importa e se o piso de dados do paper é real.

### S3 — Clusterização de tarefas

**Paper.** Clustering não supervisionado sobre prompts e ações do agente para identificar padrões recorrentes, que definem as tarefas candidatas à especialização.

**Executamos.** Embedding dos prompts inteiros (o que um cliente teria), número de clusters escolhido automaticamente (o cliente não sabe quantas tarefas tem), e validação por pureza contra as categorias do benchmark, que ficam gravadas nos logs mas escondidas do passo. Dois pipelines rodam: o "puro", que usa os clusters descobertos, e o "oráculo", que usa as categorias reais.

**Ablação.** Um único SLM treinado em todos os logs misturados, sem clusterização; e a diferença puro–oráculo.

**Reproduz.** Primeiro teste de se o clustering recupera a estrutura de tarefas e de quanto os erros dele custam lá na frente.

### S4 — Seleção de SLMs

**Paper.** Para cada tarefa, escolher candidatos por capacidades inerentes, desempenho em benchmarks relevantes, licença e footprint de implantação.

**Executamos.** Candidatos nas três faixas de tamanho, com critérios de licença, contexto e footprint (D12). Três braços de seleção: (a) o modelo mais popular da faixa, sem avaliação; (b) seleção por scores públicos de benchmark, que é o que o paper prescreve; (c) seleção pela avaliação nos próprios logs. Um candidato por faixa avança para o S5, para revelar a curva tamanho×qualidade após a especialização.

**Ablação.** (a) contra (b) contra (c).

**Reproduz.** Mede se o seletor muda o resultado, e testa a B2.

### S5 — Especialização

**Paper.** Preparar um dataset por tarefa a partir dos dados curados e fazer fine-tune do SLM; adaptação eficiente de parâmetros para reduzir custo; fine-tune completo se os recursos permitirem; destilação do LLM quando útil.

**Executamos.** Adaptação eficiente de parâmetros por cluster no SLM selecionado, com curva de dados (500 / 2k / 5k / tudo) e um braço de formato (com e sem decoding restrito). Destilação por logits só se professor e aluno compartilharem o mesmo tokenizador (D10). A ablação roda no melhor SLM e num único tamanho de dados; a curva completa fica só no braço principal.

**Ablação.** O SLM selecionado zero-shot contra o mesmo SLM especializado.

**Reproduz.** É onde V1 e A9 se decidem. Provavelmente é o passo de maior ganho; a PoC existe para confirmar ou negar isso.

### S6 — Iteração e refinamento

**Paper.** Retreinar periodicamente os SLMs e o modelo roteador com novos dados, voltando ao S2 ou ao S4 conforme o caso.

**Executamos.** Definimos o que o paper só nomeia. O **router** é o fallback: o SLM atende; se a chamada sai inválida contra o schema, o turno vai para o LLM de produção. A taxa de fallback é varrida de 0% a 100%. A **iteração** é simulada: uma categoria do benchmark fica de fora do treino como "distribuição nova"; medimos a queda, depois entram 200 e 500 logs dela e medimos a recuperação.

**Ablação.** SLM sozinho contra SLM com fallback.

**Reproduz.** O loop que o paper apenas nomeia, e o sistema misto do A6.

---

## 6. Estudos de caso (Apêndice B do paper)

**Paper.** Estima, sem experimento, que 60% das chamadas do MetaGPT, 40% do Open Operator e 70% do Cradle poderiam ser atendidas por SLMs especializados. O padrão é o mesmo nos três: o SLM serve para código rotineiro, respostas em template, parsing e roteamento de comandos e fluxos repetitivos; o LLM fica com raciocínio arquitetural, planejamento adaptativo e resolução de erros não estruturados.

**Executamos.** O análogo medido dessas estimativas: a **fração substituível**, definida como a maior fração de chamadas atendida pelo SLM, sem fallback, tal que a qualidade do sistema misto iguala a do LLM de produção dentro do intervalo de confiança. Sai da varredura de fallback do S6.

**Reproduz.** O número-manchete da PoC, comparável às estimativas do apêndice.

---

## 7. Protocolo de medição

Esta é a única seção sem par no paper, porque o paper não tem avaliação. Tudo aqui é fixado **antes** de qualquer resultado. O que se fixa é como medir; o tamanho do efeito não é fixado (D3): não há limiares de "bom o bastante", e todo resultado é reportado como curva ou fronteira.

**Verdade de referência.** O gabarito do benchmark, como ele o define (comparação estrutural ou execução). A fidelidade ao professor é reportada separadamente: imitar o professor não é acertar.

**Métricas, para cada braço e cada categoria de dificuldade:**

- acurácia, como o benchmark a define;
- taxa de chamadas inválidas contra o schema;
- custo por mil chamadas;
- latência p50 e p95.

**Splits e contaminação.** Os logs de treino vêm de inputs distintos do benchmark. O benchmark não entra no treino em nenhum braço. As categorias vivas do benchmark, criadas para resistir a contaminação, são tratadas com esse cuidado.

**Ruído.** Cada configuração roda com repetições suficientes para um intervalo de confiança; diferenças dentro do intervalo são reportadas como empate.

**Modelo de custo.**

- Modelo via API: preço público por token.
- Modelo local: custo-hora da GPU de referência dividido pelo throughput medido.
- Utilização: custo recalculado a 20%, 50% e 100% (CA3).
- Custo fixo: horas de engenharia e re-treinos periódicos entram no payback (CA4).
- Payback em três volumes de referência: pequeno, médio e grande (D18).

**Registro de esforço, por passo do algoritmo.** Horas humanas; horas de GPU; gasto com API; o que foi feito automaticamente e o que exigiu julgamento humano; o que precisaria mudar para uma segunda tarefa. Esse registro separa produto (o que roda sozinho quando se troca de tarefa) de serviço (o que exige gente a cada vez).

**Entregáveis.**

1. O gráfico principal: acurácia no benchmark contra custo por mil chamadas, com o LLM de produção, o SLM zero-shot, o SLM especializado e a curva do sistema misto.
2. A tabela de ablação: contribuição e esforço de cada passo S1–S6.
3. A fração substituível medida (Seção 6).
4. O registro de esforço.
5. O relatório, com a análise dos erros do SLM.

**Os três resultados possíveis, todos válidos.**

- O SLM iguala o LLM a uma fração do custo: V1–V3 confirmadas para tool calling.
- O gap fecha só com especialização e resiste nas categorias de múltiplos turnos: a posição vale na forma estreita que o paper defende (A6, A11).
- O SLM não chega: a posição não se sustenta nesta tarefa, e o motivo fica documentado.

---

## 8. Fora de escopo

Continuam sendo opinião do paper, sem teste nesta PoC:

| Item                                 | Razão                                                                                                           |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| A2 (iv), uso eficiente de parâmetros | Exige instrumentação interna dos modelos                                                                        |
| A3, democratização                   | Não testável numa PoC                                                                                           |
| A8, arquitetura por tamanho          | Experimento à parte; só um ponto de dado opcional (D16)                                                         |
| A10, compute em inferência           | Opcional (D15), pelo custo em latência                                                                          |
| A12, A13                             | Tendências de mercado                                                                                           |
| AV3                                  | Não testável; o registro de esforço é o único dado                                                              |
| B1, B3                               | Não testáveis                                                                                                   |
| Edge como padrão                     | Opcional (D17)                                                                                                  |
| Modelos de visão-linguagem           | O paper os menciona apenas como extensão                                                                        |
| Afirmações ambientais                | Sem instrumentação para medir                                                                                   |
| Domínio, língua e contexto longo     | A tarefa é tool calling em inglês; um "segundo cliente" em outro domínio é trabalho futuro, com o mesmo harness |
| Produção                             | É um benchmark, não um sistema em escala de cliente                                                             |

---

## 9. Regras e registro de decisões

### Regras

- Só dados públicos. Nenhum dado ou material de terceiros entra no projeto.
- O benchmark de teste não entra no treino em nenhum braço.
- Saídas do LLM de produção nunca viram alvo de treino. O professor é sempre um modelo cuja licença ou termos permitem esse uso.
- O projeto é público, nasce antes de qualquer contrato e é de autoria única. Os textos finais (relatório, memo) são escritos pelo autor; as sessões de desenvolvimento entregam estrutura, código e revisão.
- Esta spec não nomeia modelos, provedores nem bibliotecas; esses vão em configuração.

### Decisões fixadas

| #   | Decisão                                                                                            |
| --- | -------------------------------------------------------------------------------------------------- |
| D1  | Tarefa única: tool calling, medida no BFCL                                                         |
| D2  | Um workload, um benchmark; sem suíte multi-tarefa                                                  |
| D3  | Sem limiares; protocolo fixado antes, tamanho do efeito não                                        |
| D4  | Três papéis abstratos (LLM de produção, professor, SLMs), escolhidos em configuração               |
| D5  | Raciocínio do professor fora dos logs                                                              |
| D6  | LLM de produção fechado e só avaliação; professor com licença que permite treino                   |
| D7  | Esquema de log neutro entre provedores, com versão do modelo em cada registro                      |
| D8  | Ablação "tira um de cada vez" sobre o pipeline completo; no melhor SLM, num único tamanho de dados |
| D9  | Dificuldade medida nas categorias vivas e de múltiplos turnos do benchmark                         |

### Decisões abertas

| #   | Decisão                                          | O que decide                                                                                |
| --- | ------------------------------------------------ | ------------------------------------------------------------------------------------------- |
| D10 | O professor concreto                             | Licença, posição no leaderboard do benchmark, tokenizador compartilhado com algum candidato |
| D11 | O LLM de produção concreto                       | O que clientes usam de fato                                                                 |
| D12 | A lista de candidatos SLM                        | Licença, contexto, footprint, três faixas; reconferir versões no dia                        |
| D13 | O dataset de inputs para os logs                 | Licença e distância do benchmark                                                            |
| D14 | A GPU de referência                              | Representar inferência realista, não a mais cara                                            |
| D15 | Incluir o braço A10                              | Tempo disponível                                                                            |
| D16 | Incluir um candidato de arquitetura híbrida (A8) | Disponibilidade na faixa                                                                    |
| D17 | Rodada em máquina de consumo (WD1, A2 iii)       | Tempo disponível                                                                            |
| D18 | Os três volumes de referência do payback         | Cenários plausíveis de cliente                                                              |

### Cronograma

Três fases em cerca de duas semanas:

1. **Workload, logs, curadoria e zero-shot** — S1, S2 e a primeira leitura de A1 com três candidatos.
2. **Clusterização, seleção e especialização** — S3, S4, S5 com todos os candidatos, curvas e medição de custo e latência.
3. **Router, ablação e relatório** — S6, os braços de ablação, o modelo de custo completo, o registro de esforço e o relatório.

### Glossário dos rótulos do paper

- **WD1, WD2** — definições de SLM e LLM (Seção 2.1)
- **V1–V3** — as três visões (Seção 2.2)
- **A1–A7** — argumentos de sustentação (Seção 3)
- **AV1–AV3** — visões alternativas (Seção 4)
- **CA1–CA4** — contra-argumentos que sustentam as visões alternativas (Seção 4)
- **A8–A13** — réplicas do paper às visões alternativas (Seção 4)
- **B1–B3** — barreiras à adoção (Seção 5)
- **S1–S6** — passos do algoritmo de conversão (Seção 6)
- **D1–D18** — decisões deste projeto (esta seção)
