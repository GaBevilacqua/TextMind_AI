from rouge_score import rouge_scorer

ref = "Mesmo com a alta de 15% no dólar em 2013, os brasileiros gastaram no exterior um recorde de US$ 25,34 bilhões, 14% a mais que em 2012. O aumento se deve à continuidade do crescimento da renda e do emprego no Brasil, além de preços atrativos em outros países. Para conter esses gastos, o governo elevou o IOF de 0,38% para 6,38% em cartões e transações em moeda estrangeira. Apesar disso, o Banco Central prevê novo aumento de gastos em 2014, ainda que em ritmo menor. Desde 2006, as despesas no exterior crescem fortemente. Em 1999, após a maxidesvalorização cambial, os gastos recuaram. Só voltaram a ultrapassar US$ 5 bilhões anuais em 2006. A tendência de crescimento se consolidou nos anos seguintes."
gen = "A alta de 15% no dólar em 2013, a maior dos últimos cinco anos e responsável por encarecer passagens e hotéis cotados em moeda estrangeira, não impediu que os gastos de brasileiros no exterior crescessem e batessem um novo recorde histórico. Segundo números divulgados pelo Banco Central nesta sexta-feira (24), as despesas de brasileiros lá fora somaram US$ 25,34 bilhões (considerando a cotação da moeda norte-americana nesta sexta, seriam R$ 61,14 bilhões) em todo o ano passado, o que representa um crescimento de 14% sobre 2012 – que era o recorde anterior, com gastos de US$ 22,23 bilhões. O aumento dos gastos no exterior está relacionado, segundo economistas, à continuidade dos crescimentos do emprego e da renda no Brasil, mesmo com um ritmo menor de expansão, e também aos baixos preços de produtos em alguns países.Alta do IOF No fim do ano passado, porém, o governo brasileiro adotou medidas para tentar conter esse tipo de gasto."

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
scores = scorer.score(ref, gen)

for metric, result in scores.items():
    print(f"\n🔹 {metric.upper()}:")
    print(f"  - Precision: {result.precision:.4f}")
    print(f"  - Recall:    {result.recall:.4f}")
    print(f"  - F1-score:  {result.fmeasure:.4f}")
