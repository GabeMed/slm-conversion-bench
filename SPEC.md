# SPEC v2 — slm-conversion-bench

**Execução empírica de "Small Language Models are the Future of Agentic AI"**
(Belcak et al., NVIDIA Research, arXiv 2506.02153, v3 de 22/09/2026)

Versão 2, de 30/09/2026. Substitui a v1, que continua preservada em `SPEC-v1.md`. A seção 11 diz o que mudou e por quê.

## Como ler este documento

**O paper.** É um *position paper*. Afirma que SLMs são o futuro da IA agêntica, sustenta a posição com argumentos e propõe um algoritmo de conversão LLM→SLM (S1–S6), mas não contém nenhum experimento. Os "estudos de caso" do Apêndice B são estimativas.

**O que esta POC faz.** Executa o algoritmo do paper de ponta a ponta num agente open-source real, com o protocolo fixado antes de qualquer resultado. A pergunta central é a da v1: **descobrir se melhora, quanto melhora e qual passo do algoritmo entrega o ganho**. Em termos operacionais:
- quais chamadas do agente um SLM especializado assume **sem perder qualidade** frente ao LLM que o agente usa;
- se isso vale também frente à **alternativa mais barata que não exige treino**;
- a que **custo por tarefa correta**.

**Alcance da pergunta "qual passo entrega o ganho".** No núcleo, a contribuição é **medida** para o S5 (especialização: B3 × B4) e para o S6 (alocação: B4 × B5). S1–S4 são **executados e descritos**, mas a contribuição deles não é medida no núcleo (tabela da seção 5).

**Rótulos.** V (visões), A (argumentos), AV (visões alternativas), CA (contra-argumentos), B (barreiras), S (passos do algoritmo) e WD (definições) são os do paper. D são decisões nossas (seção 10). Toda citação ao paper traz a página do PDF da v3.

**Regra de nível.** Esta spec não nomeia modelos, provedores nem bibliotecas. Isso vai para a configuração e para as sessões de desenvolvimento. Aqui, "LLM de produção", "professor", "alternativa barata" e "SLMs" são papéis.
- **Exceção declarada:** o agente (CHESS) e o benchmark (BIRD e Arcwise-Plat-SQL) são nomeados, porque são o **objeto** do teste, não uma escolha de implementação.

**Formato de cada item.** O documento responde três perguntas: **o que o paper diz**, **o que executamos** e **o que isso reproduz**. Quando um item não é testável nesta POC, isso é dito.

---

## 1. Posição (Seção 2 do paper)

### Definições

**Paper (p.2).**
- **WD1:** SLM é um modelo que cabe num dispositivo eletrônico de consumo e faz inferência com latência prática para servir as requisições agênticas de um usuário.
- **WD2:** LLM é um modelo que não é SLM.
- Em 2025, os autores dizem que ficariam confortáveis em considerar SLM a maioria dos modelos abaixo de 10 bilhões de parâmetros.

**Executamos.** Adotamos WD1 e WD2.
- Os SLMs candidatos saem da triagem do S4 (seção 4), dentro de ~10B.
- Uma rodada no dispositivo de consumo do autor é opcional, e entra como dado para honrar a WD1 literalmente.

### As três visões

**Paper (p.2–3).**
- Os SLMs são (V1) suficientemente poderosos, (V2) inerentemente mais adequados e (V3) necessariamente mais econômicos para a vasta maioria das invocações de LM em sistemas agênticos.
- Os autores dizem que não fazem recomendação nem impõem obrigação: *"We do not make a recommendation or try to impose an obligation"* (p.2).
- Atribuem a leitura como "Humean moral ought" a outros: *"represents to many… a Humean moral ought"* (p.3).

**Esta POC** não testa o valor. Testa as premissas empíricas:

| Visão | O que o paper afirma | A medida que a fecha nesta POC |
|---|---|---|
| **V1** Suficiência | SLMs dão conta dos "errands" de linguagem das aplicações agênticas | Acurácia de execução do agente com o SLM especializado × com o LLM de produção, e o contraste com o SLM zero-shot (seção 5) |
| **V2** Adequação operacional | SLMs são mais adequados a sistemas agênticos | Não é medida direta. O paper a sustenta com A3–A7 (p.5–6); aqui entram A3 (tempo e custo de adaptação), A5 (formato), A6 (alocação heterogênea) e A7 (logs como dados). A4 entra pela distribuição de chamadas. |
| **V3** Economia | SLMs são mais econômicos pela virtude do tamanho | Custo por consulta correta, com throughput medido, utilização de 20/50/100% e custo fixo (seção 6.6) |

---

## 2. O workload: o agente e o benchmark

**Paper.**
- A Fig. 1 (p.4) distingue *language model agency* (o LM orquestra as ferramentas) de *code agency* (um controller chama LMs).
- O S1 manda logar as chamadas **não-HCI** do agente (p.9).
- O S3 cita como tarefas típicas *"intent recognition, data extraction, summarization of specific document types, or code generation with respect to tools available to the agent"* (p.9).
- O Apêndice B reserva ao LLM, entre outras coisas, a resolução de erros não estruturados (p.16–17).

**Executamos.** O agente open-source **CHESS**, no fluxo IR → SS → CG, rodando nas perguntas do benchmark **BIRD**.
- É *code agency*: um controller chama LMs em chamadas estreitas, e nenhuma delas conversa com um usuário.
- Tem vários tipos de chamada, incluindo **uma que o paper reserva ao LLM**: o reparo do SQL depois de um erro.

**Call sites** (tipos de chamada de LM do agente):

| Call site | O que faz | Previsão do paper, registrada antes |
|---|---|---|
| escolha da próxima ferramenta | o agente decide o próximo passo | incerta |
| extração de palavras-chave | extração | SLM |
| filtro de coluna | classificação binária por coluna (alto volume) | SLM |
| seleção de tabelas | seleção estruturada | SLM |
| seleção de colunas | seleção estruturada | SLM |
| geração de SQL candidato | geração de código para a ferramenta (o banco) | incerta |
| reparo do SQL após erro ou resultado vazio | *unstructured error resolution* | **LLM** |

**Não são chamadas de LM:** a recuperação de contexto e de entidades. Não entram na conversão.

**Dados:**
- **Logs e treino:** as 1.034 perguntas do BIRD dev que não estão no Mini-Dev, nos 11 bancos do dev.
  - ~200 delas ficam reservadas para **calibração**: piloto, seleção e limiares.
  - As ~834 restantes geram os logs.
- **Teste:** o **Arcwise-Plat-SQL**: **498 perguntas** do Mini-Dev do BIRD, usadas **como estão no arquivo** (pergunta, evidência e SQL gold corrigido).
  - Em ~81 perguntas e ~68 evidências, o texto foi reescrito em relação ao Mini-Dev, e o gold corresponde ao texto reescrito. O schema é o original.
  - As ids 119 e 120 (ausentes do Plat-SQL) ficam fora de treino e de teste.
  - Os números **não são comparáveis** com EX publicado no Mini-Dev.
  - É tocado **uma vez** por configuração congelada.
- **Correspondência treino × teste:**
  - **iguais:** o agente, os 11 bancos e a fonte das perguntas;
  - **diferentes:** as perguntas são disjuntas, e o teste usa o texto e o gabarito do Plat-SQL;
  - **diferença conhecida:** o Mini-Dev foi montado com mistura de dificuldade 30/50/20 (simple/moderate/challenging), então os resultados são **estratificados por dificuldade**.
- **Verdade:** acurácia de execução (EX) do SQL final contra o gold corrigido, medida por um **avaliador próprio**:
  - pareia por `question_id`;
  - tem timeout;
  - executa predição e gold no mesmo momento, o que resolve os golds que dependem da data.

  A fidelidade ao professor é reportada à parte, porque imitar o professor não é acertar.

**Correção do agente, por decisão do autor (30/09/2026).** No CHESS publicado, a geração e o reparo de SQL usam o schema **completo**, e a saída do seletor de schema não chega a eles (issues #31 e #34 do repositório do CHESS). A POC aplica a correção sugerida na issue #34: a geração e o reparo passam a usar o **schema selecionado**. Com isso, todo call site é consequente, como no desenho do paper do CHESS. A correção é declarada no relatório.

**Ajustes no harness do agente,** para não contaminar a comparação:
- nenhum fallback automático para outro modelo;
- re-tentativas por erro de parse limitadas e **contadas no custo**;
- a escolha da próxima ferramenta **logada como call site próprio**;
- parâmetros de geração adaptados ao que cada modelo aceita, **registrados na configuração**.

**Reproduz.** O teste que o Apêndice B não fez: num agente real, quais chamadas um SLM especializado assume.

---

## 3. Papéis e braços

**Paper.** No S1 e no S5 (p.9), o que se loga e o que o SLM aprende a imitar é o LLM que o agente já usa. O professor **é** o LLM a ser substituído.

**Papéis:**
- **LLM de produção = professor.** O LLM generalista que o agente usa.
  - É um open-weight cuja **licença de pesos permite usar as saídas para treinar outros modelos**, acessado por um provedor cujos termos também o permitem. As duas coisas são conferidas na configuração, com link e data.
  - Critério de escolha: o open-weight mais alto num índice agregado independente, com uma reserva operacional só para falha do provedor. A configuração registra qual foi.
- **Alternativa barata sem treino.** Um generalista open-weight mais barato que o LLM de produção, com o **mesmo orçamento de otimização**: prompt com exemplos (few-shot) e cache.
- **Versão de chamada única.** Text-to-SQL numa chamada só, sem o agente, com o LLM de produção e com a alternativa barata. É "o que se faria sem agente".
- **SLMs candidatos.** Dois modelos saídos da triagem de mesa do S4, dentro de ~10B (WD1), com licença que permite uso comercial e fine-tuning (seção 4).

**Braços do núcleo:**

| Braço | O que responde |
|---|---|
| **B0** LLM de produção no agente | teto de qualidade e custo de referência |
| **B1** Alternativa barata no agente | a alternativa trivial: trocar para um modelo mais barato sem treinar |
| **B2** Chamada única (produção e barata) | vale ter o agente? Fica no núcleo porque é barato (uma chamada por pergunta) e porque, se a chamada única igualar o agente, a conversão por call site perde o objeto |
| **B3** SLM zero-shot no agente | o ganho da especialização (contraste com B4) |
| **B4** SLM especializado em todos os call sites | V1 na forma mais forte desta POC |
| **B5** Alocação por cluster | cada cluster servido pelo motor mais barato que passa na não-inferioridade (seções 4, S6, e 6) |

**Extensões, nesta ordem, cada uma só depois do núcleo fechado:**
1. troca de um cluster por vez;
2. no call site de geração de SQL, par professor × gabarito (o BIRD train tem SQL gold), que separa a fonte do rótulo da capacidade;
3. escalada dinâmica ao LLM de produção quando o SQL falha na execução;
4. decoding restrito ao formato e um segundo generalista aberto sem treino;
5. mesmos bancos × bancos novos (treinar com o BIRD train e testar no mesmo teste), que mede o valor de "conhecer os bancos";
6. uma sonda multi-turn, só de inferência, num benchmark conversacional;
7. um SLM já pronto para text-to-SQL como referência sem treino;
8. um segundo professor open-weight;
9. um braço com LLM fechado de fronteira, aberto a contribuidores (seção 8);
10. rodada zero-shot ampla na calibração (6–8 candidatos da triagem do S4), para testar a B2.

**Regra de conformidade.** Saídas de qualquer modelo cujos termos proíbam treinar com elas ficam numa área separada **não-treinar**, e o pipeline de treino recusa essa área.

---

## 4. O algoritmo de conversão (Seção 6 do paper): S1–S6 executados literalmente

**A regra.** Todos os passos rodam no braço principal, na ordem do paper, e **o resultado de cada um alimenta o seguinte**. A única parte que pode ser cortada é a **volta de re-treino do S6**, e o corte é declarado (seção 7.3). O roteador do S6 nunca é cortado. Um passo que não se aplica a dados públicos ainda **é executado** e tem o efeito registrado; o desvio é declarado. Executar um passo não é fazer ablação dele: a contribuição de cada passo é medida contra alternativas nomeadas (seção 5), ou declarada não testada.

### S1 · Coleta segura de uso

**Paper (p.9).** *"deploying instrumentation to log all non-HCI agent calls, capturing input prompts, output responses, contents of individual tool calls, and optionally latency metrics… encrypted logging pipelines with role-based access controls and anonymize all data with respect to its origins before storage."*

**Executamos.**
- Um logger na interface de chamada, o *listener* da Fig. 1, grava por chamada:
  - prompt;
  - resposta;
  - SQL executado e resultado da execução;
  - latência;
  - tokens de entrada, cache e saída;
  - modelo e versão;
  - id do call site, **oculto do S3**.
- O esquema é neutro entre provedores.
- Há criptografia em repouso, acesso restrito e anonimização de origem.
- O raciocínio interno do professor, quando exposto, fica fora do log (D5).
- O que é logado: as trajetórias do professor (= LLM de produção) nas perguntas de treino.

**Desvio.** Dados públicos, sem usuários reais. Os controles de segurança rodam, mas não têm o que proteger.

### S2 · Curadoria e filtragem

**Paper (p.9).** *"10k-100k examples being sufficient for fine-tuning of small models as a rule of thumb… remove any PII, PHI… can be often automatically paraphrased to obfuscate named entities and numerical details."*

**Executamos:**
- **volume por cluster,** reportado contra a **regra de bolso** de 10k–100k (não é um piso). Os clusters de baixo volume ficam abaixo, e isso é dado sobre a regra;
- **mascaramento de dados sensíveis,** executado, com o nº de detecções reportado;
- **filtro de sucesso** com um sinal disponível em produção, nunca o gabarito: o SQL executa sem erro, o resultado não é vazio e o formato é válido;
- **deduplicação** exata e por quase-duplicata.

**Desvio.** A paráfrase de entidades e números **não é aplicada**. Em text-to-SQL, os valores e nomes fazem parte da resposta correta, e parafraseá-los quebraria a tarefa. Registrado como achado sobre o S2 em tarefas estruturadas.

### S3 · Clustering de tarefas

**Paper (p.9).** *"Employ unsupervised clustering techniques on the collected prompts and agent actions to identify recurring patterns."*

**Executamos.**
- Clustering não supervisionado sobre prompt + ação, com número de clusters automático e **sem ver o id do call site**.
- **Os clusters descobertos definem as unidades de especialização** de S4 e S5.
- Validação: concordância (ARI) entre clusters e call sites.
  - **ARI ≈ 1:** o S3 é trivial em *code agency*, e isso é reportado assim.
  - **Senão:** o pipeline segue com os clusters descobertos, e o custo do erro de clustering aparece na qualidade final.
- **Atribuição de uma chamada nova a um cluster, em tempo de execução.** Usa **só o que existe antes da resposta**: o prompt, que no reparo já traz o SQL que falhou e a mensagem de erro.
  - Regra: cada chamada vai para o centróide mais próximo, calculado sobre o prompt apenas.
  - A taxa de atribuição correta, contra os clusters do S3, é medida na calibração e reportada.
  - A ação do professor ajuda a **formar** os clusters, mas nunca entra na **atribuição**.

### S4 · Seleção de SLMs

**Paper (p.9).** *"For each identified task, select one or more candidate SLMs. Criteria for selection include the SLM's inherent capabilities (e.g., instruction following, reasoning, context window size), its performance on relevant benchmarks for the task type, its licensing, and its deployment footprint (memory, computational requirements). Models of Section 3.2 serve as good starting candidates."* O texto remete à seção 3.2, mas a lista de modelos está na 3.1 (A1, p.3–4).

**Executamos.**
- **(a) Triagem de mesa documentada.**
  - Parte da lista do próprio paper (A1), nas versões atuais, mais os candidatos atuais dentro de ~10B (WD1).
  - Aplica os quatro critérios, com o resultado de cada candidato numa tabela publicada no relatório:
    1. capacidades: seguir instruções, raciocínio, e janela de contexto ≥ o maior prompt observado no S1 para o cluster;
    2. benchmarks públicos relevantes para o tipo de tarefa;
    3. licença que permite uso comercial e fine-tuning;
    4. footprint: cabe numa GPU de referência.
- **Partir da lista do paper não dá preferência a ela.** A escolha é só pelos critérios.
- **(b)** Saem **2 candidatos** para a checagem zero-shot na calibração, **nunca no teste**.
  - Primeiro, os filtros obrigatórios: licença, janela, footprint e suporte no servidor de inferência.
  - Depois, a ordem é pela **evidência independente no tipo de tarefa** (tool calling, saída estruturada, SQL). Número auto-reportado só desempata.
- **(c)** O melhor segue para o S5. Seguem 2 se o orçamento permitir, como o S5 admite (*"fine-tune the chosen SLMs"*).
- **Desempate pré-registrado:** maior acurácia na avaliação por call site na calibração; empatando, menor footprint.

**Desvio.** O paper pede "one or more" candidatos por tarefa; a POC usa os mesmos 2 candidatos para todos os clusters. A B2 (seção 5) não é testável com 2 candidatos e fica para a extensão 10.

### S5 · Fine-tuning especializado

**Paper (p.9).** *"fine-tune the chosen SLMs… PEFT techniques such as LoRA or QLoRA… knowledge distillation, where the specialist SLM is trained to mimic the outputs of the more powerful generalist LLM."*

**Executamos.**
- Um adaptador eficiente em parâmetros por cluster, sobre a base escolhida no S4, todos servidos juntos.
- **Destilação por sequência:** SFT nas saídas curadas do professor.
- A destilação por logits só entra se professor e aluno compartilharem o tokenizer e o professor expuser logits. Do contrário, fica declarada como não aplicada.
- O tempo e o custo de cada adaptador são registrados. É o *"overnight rather than over weeks"* do A2 (p.5).

### S6 · Iteração e refinamento

**Paper (p.9).** *"One may retrain the SLMs and the router model periodically with new data… returning to step S2 or step S4 as appropriate."* O paper apresenta a iteração como opcional ("One may") e **pressupõe um roteador sem defini-lo**.

**Executamos.**
- **Roteador (nunca cortado):** é **decisão desta POC**, necessária para testar o A6 (sistema heterogêneo). Não é obrigação do texto do S6. A POC usa o roteador mais simples compatível com o paper, uma **política por cluster** (braço B5).
  - A chamada é atribuída ao cluster pela regra do S3 (só o prompt).
  - **Clusters com gabarito por chamada** (geração e reparo de SQL): o motor mais barato cuja avaliação por call site passa na não-inferioridade na calibração, com margem Δ/2 e n suficiente (6.4).
  - **Clusters sem gabarito por chamada** (palavras-chave, filtro de colunas, seleção de tabelas e de colunas, escolha de ferramenta): o motor mais barato cuja **concordância com o professor** na calibração atinge o limiar pré-registrado (6.4). É **proxy**, declarado como tal, e não sustenta alegação por cluster.
  - **Cluster sem evidência suficiente fica com o LLM de produção.**
  - Um roteador treinado (o "router model" do paper) é extensão.
- **Uma volta do laço (a única parte cortável do S6, declarada se cortada):**
  1. análise de erro na calibração;
  2. volta ao S2 (ajustar o filtro) ou ao S4 (trocar a base) no cluster mais fraco;
  3. um re-treino;
  4. nova avaliação na calibração.

  O teste só é tocado no fim, com tudo congelado.

**Desvio.** O re-treino "periódico, com dados novos ao longo do tempo" não é testável em 2 dias.

---

## 5. Mapa das afirmações: o que confirma, o que refuta, o limite

Cada linha foi fixada antes de qualquer resultado. "Confirma" e "refuta" usam a margem Δ da seção 6. Resultado sem poder é "inconclusivo", não confirmação.

| Afirmação (paper) | Comparação | Confirma se | Refuta se | Limite do que se conclui |
|---|---|---|---|---|
| **V1 / A1:** SLMs bastam para as chamadas de agentes (p.3–4) | **ponta a ponta** (n = 498): B4 × B0 e B5 × B0, e o contraste B3 × B4; **por call site** (6.2) só nos clusters com gabarito (geração e reparo) | B5 (e/ou B4) não-inferior a B0 ponta a ponta | B4 e B5 abaixo de −Δ ponta a ponta | um agente, um domínio (SQL), *code agency*; nos clusters sem gabarito não há alegação por cluster, só a concordância com o professor como proxy |
| **A4 / A11:** as chamadas são estreitas e as subtarefas simples (p.5, p.7) | fração substituível por **chamada, token e custo** | fração alta também por token e custo | fração alta só por chamada (as baratas) e baixa por custo | idem |
| **Apêndice B:** o LLM fica com "unstructured error resolution" (p.16) | **avaliação por call site** no cluster de reparo: o EX do SQL reparado pelo SLM × pelo LLM, com o mesmo contexto | o SLM perde no reparo e passa nos clusters de rotina | o SLM empata no reparo (a partição é conservadora) **ou** perde na rotina | um call site de reparo; se as chamadas de reparo não sustentarem Δ_cluster ≤ 5 p.p., "não testável" |
| **A5:** formato único com SLM treinado é preferível (p.6) | validade de formato por call site: B4 × B0 | B4 ≥ B0 sem truque de decoding | B4 pior | os formatos deste agente |
| **A6:** sistemas heterogêneos (p.6) | B5 × B4 × B0 × B1 | B5 não-inferior a B0 e mais barato que o melhor sem treino | B5 não vence o melhor braço sem treino | política por cluster desenhada por nós; o paper diz "can be" e admite sistema só de SLMs (Fig. 1) |
| **A7:** os logs do agente viram dados (p.6) | B4 treinado nos logs curados do professor (+ extensão 2) | logs ≈ gabarito | gabarito ≫ logs | professor aberto |
| **V3 / A2:** SLM de 7B é 10–30× mais barato "in latency, energy consumption, and FLOPs" que LLM de 70–175B (p.4) | custo por consulta correta, B4/B5 × B0 × B1 | ≥3× mais barato que o melhor braço sem treino que passa em V1 | <3× | a medida é custo em dólar por consulta correta, **não** o 10–30× de FLOPs; preço de GPU de mercado |
| **AV2 / CA3–CA4:** a escala centralizada pode ser mais barata (p.7–8) | o mesmo, contra B1 com cache, a 20/50/100% de utilização e com custo fixo | — | B1 com cache ≤ SLM por consulta correta na utilização realista (vitória da AV2) | idem |
| **A2 (fine-tuning agility) / A3:** adaptar é rápido e barato (p.5) | tempo e custo por adaptador; o registro de esforço | — | — | medida descritiva |
| **B2:** benchmarks generalistas guiam mal a seleção (p.8) | **não testável no núcleo**: com 2 candidatos, comparar rankings não é informativo. Na extensão 10, rodada zero-shot ampla (6–8 candidatos) | (extensão 10) os rankings divergem | (extensão 10) os rankings coincidem | só com a extensão 10 |
| **S3:** o clustering descobre as tarefas (p.9) | ARI clusters × call sites | — | — | em *code agency* tende a ser trivial |
| **AV1:** LLM da mesma geração sempre vence (disputa a V2, p.7) | B0 × B4 | — | — | nossos modelos **não** são "da mesma geração" no sentido do paper; não é teste direto da AV1 |

**Fração substituível.**
- É operacional e sem oráculo: a maior fração de chamadas, tokens e custo servida por SLM no B5 com a não-inferioridade satisfeita.
- O teto de oráculo aparece à parte, só como referência.
- **Não é comparável** às estimativas de 60/40/70% do Apêndice B. Aquelas são frações por papel em agentes inteiros, e um deles, o Cradle, é um agente de screenshots.
- Conta **só os clusters que o B5 atribuiu ao SLM**, o que a torna conservadora.

**Contribuição de cada passo (D8).** O que é medido no núcleo, e o que é declarado não testado:

| Passo | Comparação no núcleo | Se não houver |
|---|---|---|
| S1 (coleta) | — | **não testado** como contribuição: sem logs não há especialização; volume e custo reportados |
| S2 (curadoria) | efeito do mascaramento e taxa de aprovação do filtro, descritivos | **não testado** como contribuição; filtrado × não filtrado é extensão |
| S3 (clustering) | ARI e taxa de atribuição | contribuição **não testada**; se ARI ≈ 1, trivial |
| S4 (seleção) | triagem documentada pelos 4 critérios; zero-shot dos 2 candidatos na calibração | contribuição **não testada** no núcleo (a B2 fica na extensão 10) |
| S5 (especialização) | **B3 × B4** (zero-shot × especializado), ponta a ponta e por call site | — |
| S6 (roteador e iteração) | **B4 × B5** (tudo em SLM × alocação); a volta de re-treino: calibração antes × depois | se a volta for cortada: roteador medido, iteração **não testada** |

---

## 6. Protocolo de medição

Tudo aqui é fixado **antes** de qualquer resultado no teste.

### 6.1 Pré-registro e registro do teste
- **Antes de tocar no teste,** publica-se o hash de:
  - esta spec;
  - a configuração (modelos por papel, provedores, parâmetros, seeds);
  - as listas de IDs de calibração e de teste;
  - a regra de cálculo de Δ.
- **Registro do teste:** cada configuração é avaliada uma vez, e todas as avaliações feitas são reportadas.

### 6.2 Unidades
- **Pergunta:** para acurácia (EX) e custo.
- **Chamada:** para formato e fração substituível.
- **Chamada com o contexto do professor (avaliação por call site).** Cada call site é avaliado com as mesmas entradas que o LLM de produção recebeu na execução do B0, e o SLM responde no lugar dele.
  - A saída é comparada com o gabarito quando existe: o EX do SQL gerado ou reparado.
  - Quando não existe gabarito, é comparada com a saída do professor, e isso é reportado como **fidelidade**, não como acerto.
  - É a base das conclusões por cluster. As conclusões ponta a ponta vêm das execuções completas.
- **Cluster:** para a alocação.

### 6.3 Métricas, por braço e por dificuldade
- acurácia de execução (EX);
- validade de formato por call site;
- **custo por consulta correta**;
- latência p50 e p95;
- fração substituível (chamada, token, custo);
- tempo e custo por adaptador;
- registro de esforço (6.8).

### 6.4 Estatística
- **Piloto:** ~50 perguntas da calibração medem d, a discordância pareada entre o SLM e o LLM de produção.
- **Margem:** Δ = (z₀,₉₅ + z₀,₈₀)·√(d/498).
  - **Teto: Δ ≤ 5 p.p.**
  - Se o Δ calculado passar do teto, a não-inferioridade é reportada como **"não testável com n = 498"** (só descritiva).
- **Não-inferioridade:** IC unilateral por bootstrap pareado, com a pergunta como unidade.
- **Alegações conjuntivas** ("substitui em todos os clusters de rotina") exigem que todos os testes passem, sem correção de Holm, e o poder conjunto é reportado.
- **"Inconclusivo"** conta como não confirmado, e é reportado **com o poder**, para não ser lido como refutação.
- **Seleção na calibração** (a alocação do B5) usa margem Δ/2, para não escolher motores que estão no limite.
- **Limiar de concordância** (clusters sem gabarito, no B5): ≥95% de concordância com o professor na calibração. A concordância é definida por call site (lista igual, JSON igual, decisão igual), e o limiar é fixado no pré-registro.
- **Conclusões por cluster** só existem nos clusters com gabarito (geração e reparo). Nos demais, a alegação é **só ponta a ponta** (B5 × B0 no teste).
  - Δ_cluster = (z₀,₉₅ + z₀,₈₀)·√(d_c/n_c), com n_c o nº de chamadas do cluster no teste e d_c a discordância medida no piloto.
  - Com Δ_cluster > 5 p.p., a conclusão daquele cluster é **"não testável"**.
  - No B5, um cluster sem evidência suficiente na calibração **fica com o LLM de produção**.
  - A unidade de reamostragem continua sendo a pergunta, agrupando as chamadas dela.

### 6.5 Roteador e fração substituível
- O roteador usa só informação disponível em tempo de execução: o cluster atribuído pelo **prompt** (regra do S3) e a política escolhida na calibração. A resposta do professor nunca entra na decisão.
- Qualquer roteador por escore (extensão) calibra o limiar na calibração, com perda binária.

### 6.6 Custo
- **Modelos via API:** o `usage` de cada chamada (entrada, cache, saída) × a tabela de preços oficial, datada na configuração.
  - As variantes de preço (sem cache, com desconto de lote) são **recalculadas a partir do mesmo `usage`**, sem rodar de novo.
  - Re-tentativas entram.
- **SLM:**
  - custo por pergunta = preço/hora da GPU de referência ÷ (perguntas por hora sustentadas dentro do p95 × utilização), a **20, 50 e 100%**;
  - o throughput é medido por um **teste de carga que reproduz as chamadas reais do agente**, numa GPU alugada com cobrança por segundo e com cache de prefixo ligado;
  - o custo fixo (horas de engenharia e re-treinos) entra no payback em três volumes de referência.
- **Opcional:** uma medição no dispositivo de consumo do autor (WD1), reportada à parte.
- **Extensão:** a estimativa do custo em LLMs fechados, repreçando as contagens de token, rotulada como estimativa e sem alegação de qualidade.

### 6.7 Latência justa
- O mesmo perfil de carga para todos os braços.
- Braços intercalados no tempo.

### 6.8 Registro de esforço, por passo
- horas humanas;
- horas de GPU;
- gasto de API;
- o que rodou sozinho e o que exigiu julgamento;
- o que mudaria numa segunda tarefa.

O tempo de construir o harness fica **separado** do tempo de conversão.

---

## 7. Núcleo de 2 dias, pré-condições, cortes e fallback

### 7.1 Pré-condições (manhã do Dia 1)

Cada uma com a ação de corte correspondente:

| Pré-condição | Se falhar |
|---|---|
| O agente roda uma pergunta ponta a ponta em ≤3 h | usar um controller equivalente: os mesmos call sites, com os mesmos prompts, a mesma ordem e o mesmo fluxo de dados, e o loop de reparo. O relatório declara qual foi usado. Se nem isso rodar até o fim da manhã, vai para o **fallback** (7.4). |
| O conjunto de call sites observado nos logs de treino e nas rodadas de calibração está registrado (id gravado em cada chamada) | corrigir o harness antes de seguir |
| Os IDs do Mini-Dev casam com o dev, e o gabarito corrigido está acessível | corrigir o mapeamento; sem gabarito, não há teste |
| A licença e os termos do professor e do provedor permitem treinar com as saídas | trocar pela reserva operacional |
| O gasto do piloto, projetado para o núcleo, cabe no teto da configuração | cortar extensões e reduzir o piloto antes de tocar o núcleo |
| O throughput e o tempo de treino medidos confirmam o cronograma | aplicar os cortes de 7.3 |

**Checagem durante o teste (não é pré-condição).** Nas rodadas de teste do Dia 2, uma asserção automática compara o id de cada chamada com o conjunto registrado. Se aparecer um call site fora do conjunto, a configuração é interrompida, sinalizada, e o caso vai ao relatório.

### 7.2 Núcleo (tudo tem de rodar para a POC valer)

| # | Item | Critério de aceite |
|---|---|---|
| K1 | Logs do professor nas perguntas de treino, em todos os call sites | ≥90% das perguntas com trajetória completa; volume por call site registrado |
| K2 | B0, B1, B2 e B3 no teste | as 498 perguntas avaliadas; custo e p95 medidos |
| K3 | S2 → S3 → S4 → S5: os adaptadores treinados | treino converge; formato válido ≥95% na calibração |
| K4 | B4 e B5 no teste, com o roteador do S6 (depois da volta de re-treino, ou sem ela se cortada, declarado), e a avaliação por call site | as 498 perguntas; custo por consulta correta; n e Δ_cluster por cluster |
| K5 | O relatório (seção 8), incluindo os negativos, o poder e o registro do teste | todos os itens da seção 8 |

### 7.3 Cronograma e cortes

| | Manhã | Tarde / noite |
|---|---|---|
| **Dia 1** | Pré-condições (7.1), piloto e cálculo de Δ | K1 (logs), S2 e S3; disparar os treinos do S5 para rodar à noite |
| **Dia 2** | S6 (uma volta) e K2 e K4 no teste | custo, estatística, relatório |

- **Se o tempo ou o orçamento apertarem,** cortam-se as extensões na ordem inversa da lista, e depois a volta de re-treino do S6 (declarada; o K4 roda sem ela).
- **Nunca se cortam:**
  - K2 (a comparação com o caminho sem treino);
  - K5 (o relatório);
  - o roteador do S6 (B5);
  - a avaliação por call site.

### 7.4 Fallback: um teste mais estreito, declarado como tal

Se o agente não rodar, a POC vira um teste de text-to-SQL de **chamada única**: LLM de produção × alternativa barata × SLM especializado. Ela **não** alega testar a conversão de um agente.

| Continua testável | Deixa de ser testável |
|---|---|
| V1 (numa tarefa única) | A4/A11 (a distribuição de chamadas num agente) |
| A5 (formato) | a partição do Apêndice B (o reparo) |
| A7 (logs de chamada única como dados) | A6 (a alocação heterogênea por call site) |
| V3/AV2 (custo por consulta correta) | S3 (clustering) |
| | a fração substituível por tipo de chamada |

---

## 8. Entregáveis

1. **Repositório público:**
   - código;
   - configuração (onde os modelos e provedores são nomeados);
   - o hash do pré-registro;
   - o registro do teste;
   - os logs agregados.
2. **Relatório:**
   - o gráfico principal: EX × custo por consulta correta, com B0–B5;
   - o **mapa da seção 5 preenchido**: confirma, refuta ou inconclusivo, com o poder;
   - a tabela S1–S6: o que cada passo fez, custou e mudou;
   - a **tabela da triagem do S4**: candidatos × os 4 critérios do paper, com o resultado de cada um;
   - a fração substituível por chamada, token e custo;
   - a análise dos erros do SLM;
   - as limitações;
   - uma seção de **referências externas não comparáveis**: resultados publicados de modelos nas mesmas perguntas ou no mesmo benchmark, obtidos com outro harness, outra versão ou outro gabarito, sempre rotulados assim.
3. **Pacote para contribuidores** (produzido **depois do núcleo**, fora dos 2 dias):
   - um comando por braço extra (extensões 8 e 9);
   - IDs, commit e parâmetros fixados;
   - os artefatos exigidos: logs brutos com os identificadores de requisição e o `usage` de cada chamada;
   - a atribuição a quem rodou.

   A regra de conformidade (seção 3) vale para qualquer contribuição.
4. **Envio aos autores** pelo canal que o próprio paper abriu para contribuições e críticas (p.1 e p.9), que eles se comprometem a publicar.

Os textos finais (relatório, posts) são escritos pelo autor. As sessões de desenvolvimento entregam estrutura, código e revisão.

---

## 9. Fora de escopo

| Item | Razão |
|---|---|
| A2 (parameter and embedding space utilization) | exige instrumentação interna dos modelos |
| A3 (democratização) | não testável numa POC |
| A8 (arquitetura por tamanho) e A10 (compute em inferência) | experimentos à parte |
| A12, A13, AV3, B1, B3 | tendências de mercado ou de adoção |
| Agentes conversacionais inteiros | o paper deixa esse terreno ao LLM (Apêndice B); só a sonda de inferência como extensão |
| Re-treino periódico ao longo do tempo | não cabe em 2 dias |
| Dados reais, PII real, drift real, velocidade de uma segunda conversão | fora de uma POC pública com dados de benchmark |
| Modelos de visão-linguagem, afirmações ambientais | sem instrumentação para medir |
| Outros domínios e idiomas | o mesmo harness serve para uma segunda tarefa, como trabalho futuro |

---

## 10. Regras e decisões

### Regras
- **Dados:** só dados e código públicos, com licença que permite este uso. Nenhum dado privado.
- **O teste** não entra em treino, seleção nem calibração, em nenhum braço.
- **Saídas proibidas para treino:** as de qualquer modelo cujos termos proíbam treinar com elas nunca viram alvo de treino, nem filtro, nem rótulo (área não-treinar, seção 3).
- **Projeto:** público, anterior a qualquer contrato e de autoria única.
- **Nível:** esta spec não nomeia modelos, provedores nem bibliotecas; isso vai para a configuração (exceção: o agente e o benchmark, que são o objeto).

### Decisões fixadas

| # | Decisão |
|---|---|
| D1 | Workload: o agente CHESS nas perguntas do BIRD. Logs no dev fora do Mini-Dev; teste no Arcwise-Plat-SQL (498, usado como está no arquivo) |
| D2 | Um workload e um benchmark; sem suíte multi-tarefa |
| D3 | Protocolo fixado antes, **com** margem de não-inferioridade pré-registrada (regra da 6.4). Curvas e fronteiras continuam sendo reportadas |
| D4 | Papéis abstratos, com a escolha concreta na configuração |
| D5 | O raciocínio interno do professor fica fora dos logs |
| D6 | LLM de produção = professor, open-weight com licença e termos de acesso que permitem treinar com as saídas |
| D7 | Esquema de log neutro entre provedores, com modelo e versão em cada registro |
| D8 | A contribuição de cada passo é medida contra alternativas nomeadas (seção 5 e extensões), nunca "tirando o passo" |
| D9 | Resultados estratificados por dificuldade, com todas as categorias reportadas |
| D10 | Sem LLM fechado no núcleo; o braço fechado é extensão aberta a contribuidores |
| D11 | O fallback de chamada única é declarado como teste mais estreito (7.4) |
| D12 | S4: triagem de mesa documentada pelos 4 critérios do paper, partindo da lista dele (sem preferência por ela); 2 candidatos no zero-shot; 1 (ou 2) no fine-tuning; desempate pré-registrado |
| D13 | O CHESS é corrigido para que a geração e o reparo usem o schema selecionado (issue #34 do CHESS), declarado no relatório (autor, 30/09/2026) |
| D14 | O teste é o Arcwise-Plat-SQL com 498 perguntas, usado como está no arquivo; as ids 119 e 120 ficam fora de treino e de teste; não comparável com o EX publicado no Mini-Dev (autor, 30/09/2026) |
| D15 | Nos clusters sem gabarito por chamada, o B5 escolhe pela concordância com o professor (≥95% na calibração), declarada como proxy; as alegações nesses clusters são só ponta a ponta (autor, 30/09/2026) |

### Decisões em configuração, fechadas antes da leitura zero-shot do piloto
- modelos por papel e a reserva operacional;
- provedor fixado;
- GPU de referência;
- teto de gasto;
- parâmetros de geração e de raciocínio;
- seeds.

---

## 11. O que mudou da v1 e por quê

| Problema da v1 | Onde está a correção na v2 |
|---|---|
| Treino e teste vinham de agentes diferentes (logs de um dataset aberto, teste no BFCL) | seção 2 e D1 |
| Não havia alternativa sem treino | B1 e B2 (seção 3) |
| Roteador por "chamada inválida" e fração substituível dependente de oráculo | S6, 5 e 6.5 |
| "Empate dentro do IC" usado como equivalência | 6.4 e D3 |
| A ablação "tira um de cada vez" decidia por construção | D8 e seção 5 |
| O escopo não cabia no prazo | seção 7 |

**Correções de fidelidade ao paper aplicadas:**
- o paper afirma SLM > LLM em vários pontos (p.4, p.8), não só no formato;
- 10k–100k é "regra de bolso", não piso (p.9);
- o 10–30× é latência, energia e FLOPs de 7B contra 70–175B (p.4);
- *"overnight rather than over weeks"* está no A2 (p.5);
- a V2 é sustentada por A3–A7 (p.5–6), e a AV1 disputa a V2 com modelos "da mesma geração" (p.7);
- o "ought" é atribuído "to many", e o paper não faz recomendação (p.2–3);
- o A6 diz "can be", com o LLM na raiz, e admite sistema só de SLMs (Fig. 1);
- a fração substituível não é comparável ao Apêndice B;
- o SmolLM2 começa em 125M parâmetros (p.3);
- a regra de dados foi reescrita sem a contradição com o uso de benchmark público;
- as decisões de configuração fecham antes da leitura zero-shot.

---

## Glossário dos rótulos do paper

- **WD1, WD2** — definições de SLM e LLM (seção 2.1, p.2)
- **V1–V3** — as três visões (seção 2.2, p.2)
- **A1–A7** — argumentos de sustentação (seção 3, p.3–6)
- **AV1–AV3** — visões alternativas (seção 4, p.6–8)
- **CA1–CA4** — contra-argumentos que sustentam as visões alternativas (seção 4, p.7–8)
- **A8–A13** — réplicas do paper às visões alternativas (seção 4, p.7–8)
- **B1–B3** — barreiras à adoção (seção 5, p.8)
- **S1–S6** — passos do algoritmo de conversão (seção 6, p.9)
- **D1–D15** — decisões deste projeto (seção 10)
