# Decisões de engenharia

## Escopo
Planejamento de expansão de redes com orçamento limitado. O cenário é um benchmark sintético independente.

## Entregas atuais
Ambiente de expansão em grafo; baseline guloso; MCTS com seleção UCT, expansão, rollout e retropropagação; busca exata para pequenas instâncias; rede residual com saídas de política e valor treinada por exemplos do solver.

## Evolução
Guiar MCTS por priors e valor aprendidos; comparar PPO; introduzir terreno raster real; validação hidráulica; agente adversário com restrições explícitas.

## Qualidade e colaboração
CI executa testes de comportamento. Revisões devem conferir contratos de entrada,
invariantes, divisão de dados e métricas. Evoluções de infraestrutura exigem avaliação
de custo, autorização e observabilidade. Não há recursos cloud provisionados.

## Limitações
Validar a hipótese com dados públicos licenciados ou dados autorizados antes de uso
operacional. Relatar baseline, várias sementes, intervalos de incerteza, consumo
de recursos e resultados negativos. O backlog não constitui funcionalidade implementada.
