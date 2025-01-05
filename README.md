# Inżynierka

# Co robimy
1. Robimy algorytmy które będą **klasteryzować dane**.
2. Wiemy do jakich klastrów należą niektóre z punktów.
3. Operujemy tylko na danych liczbowych.
4. Algorytm tworzy klastry tylko dla danego dataset. O to chodzi w tej transdukcji, że nie tworzy się modelu. Jak mamy klastry i pojawi się nowy punkt w danych to trzeba liczyć na nowo wszystkie klastry. 

# Linki
> #### Ogólnie, co to jest ta transdukcja:<br>
> https://en.wikipedia.org/wiki/Transduction_(machine_learning)<br><br>

> #### To wszystko co robimy można podpiąć pod semi-supervised learning:<br>
> https://arxiv.org/pdf/1307.0252<br><br>

> #### Bardzo możliwe że jednym z algorytmów to będzie zmodyfikowany K-Means:<br> 
> https://www.ire.pw.edu.pl/~pplonski/papers/PP_KZ_kmeans_icannga2013.pdf - tu zrobili coś takiego<br>
> > *UPDATE: Pewnie nie będzie to K-Means, bo z założenia tworzy on model, ale nadal K-Means można używać jako pomocnik w algorytmach innych, np: [Spectral Clustering](https://en.wikipedia.org/wiki/Spectral_clustering) (ono też oparte jest na grafach)*<br><br>

> #### Algorytmy oparte na drzewach też warto sprawdzić, bo Prof. Gągolewski napisał paper + drzewa rozpinające są z założenia transduktywne (wyglądają inaczej dla set vs. set+ jeden punkt):<br>
> https://link.springer.com/content/pdf/10.1007/s00357-024-09483-1.pdf<br><br>

> #### Graph-Based clustering (ogólniejsze niż drzewa, pojawia się problem reprezentacji danych za pomocą grafu):<br>
> https://www.cse.msu.edu/~cse802/S17/slides/Lec_20_21_22_Clustering.pdf
# Dane do testowania
Najlepiej skorzystać z gotowych dataset, do testowania, bardzo dużo jest w [Repozytorium prof. Gągolweskiego](https://github.com/gagolews/clustering-data-v1/)

# Nauka
Pewnie trzeba będzie się nauczyć NetworkX żeby z drzew korzystać; zakładam że nie musimy od początku pisać algorytmów szukających min. drzewa rozpinającego:<br>
https://networkx.org/ <br><br>
### ***JAK SIĘ PISZE BIBLIOTEKI W PYTHONIE?***<br>
Nw to się wydaje przydatne<br>
https://www.reddit.com/r/learnpython/comments/13ouob6/how_to_create_a_python_packagelibrary/
# Do ustalenia
1. Czy wiemy ile klastrów ma być w danych
2. Czy przyjmujemy założenie że otrzymamy jakiegoś reprezentanta dla każdego z klastrów
3. Czy przyjmujemy założenia o tym, ile danych (minimum/maksimum) ma być sklasyfikowanych
   
