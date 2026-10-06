# Saneamento inteligente

Planejamento de expansão de redes com orçamento limitado.

**Status: MVP de pesquisa, v0.1.0.** Estudo de caso independente inspirado
em desafios dos setores relacionados à trajetória do autor (ANA · Águas Brasil · BiomaData).
Não representa entrega contratada, resultado real ou endosso dessas organizações.
Todos os exemplos desta versão são sintéticos e reproduzíveis.

## Executar

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e .
python -m sanplan.demo --iterations 200 --seed 7
python -m unittest discover -s tests -v
```

## Implementado

Ambiente de expansão em grafo; baseline guloso; MCTS com seleção UCT, expansão, rollout e retropropagação; busca exata para pequenas instâncias; rede residual com saídas de política e valor treinada por exemplos do solver.

## Próximas etapas

Guiar MCTS por priors e valor aprendidos; comparar PPO; introduzir terreno raster real; validação hidráulica; agente adversário com restrições explícitas.

## Reprodutibilidade e avaliação

A demonstração grava `artifacts/report.json`, com parâmetros e métricas.
Resultados de dados sintéticos validam o fluxo; não demonstram desempenho em produção.
Use a mesma semente e orçamento computacional para comparar abordagens.
Consulte `docs/design.md` para decisões e limitações.

## Contribuição

Abra uma issue com objetivo e critério de aceite. Entregue mudanças pequenas,
revisadas e acompanhadas de testes de comportamento. Não inclua dados de clientes,
credenciais ou resultados sem proveniência. Licença MIT para código e dados sintéticos próprios.
