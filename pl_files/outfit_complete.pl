nn(fashion_mnist_net,[X], O, [0,1,2,3,4,5,6]) :: category(X,O).

%% OLD 
%Tshirt = 0
%Trouser = 1
%Pullover = 2
%Dress = 3
%Coat = 4
%Sandal = 5 X
%Shirt = 6
%Sneaker = 7 X
%Bag = 8
%Ankleboot = 9 X

%% NEW (we no longer use shoes)
%Tshirt = 0
%Trouser = 1
%Pullover = 2
%Dress = 3
%Coat = 4
%Shirt = 5
%Bag = 6


%%%%%%%% Problog non-built ins
member(El, [H|T]) :- member_(T, El, H).
member_(_, El, El).
member_([H|T], El,_) :- member_(T, El, H).

% once(Goal) :- call(Goal),!.

memberchk(El, L) :- once(member(El,L)).
%%%%%%%%

% Appropriate Wardrobe ------------------------------------------------------------------------------------------------------------------

% Wardrobe must include a coat or pullover.
goodForRain(Pieces,1) :- memberchk(2,Pieces).
goodForRain(Pieces,1) :- memberchk(4,Pieces).
goodForRain(Pieces,0) :- \+goodForRain(Pieces,1).

% Wardrobe must not include a tshirt or a dress
formal(Pieces,0) :- memberchk(0,Pieces).
formal(Pieces,0) :- memberchk(3,Pieces).
formal(Pieces,1) :- \+formal(Pieces,0).

% Whether the pieces int the given wardrobe make a full outfit.
fullOutfit(Pieces,1) :- hasTop(Pieces), hasBottoms(Pieces).
fullOutfit(Pieces,0) :- \+fullOutfit(Pieces,1).

hasTop(Pieces) :- memberchk(0,Pieces). 
hasTop(Pieces) :- memberchk(2,Pieces).
hasTop(Pieces) :- memberchk(5,Pieces).
hasTop(Pieces) :-  memberchk(4,Pieces).
hasBottoms(Pieces) :- memberchk(1,Pieces).
hasBottoms(Pieces) :- memberchk(3,Pieces).

getCategories([I],[C]) :- category(I,C).
getCategories([I|Is],[C|Cs]) :-
    category(I,C),
    getCategories(Is,Cs).

appropriateWardrobe(Images, Rain, Formal, Full) :- 
    getCategories(Images,Categories)
    goodForRain(Categories,Rain),
    formal(Categories,Formal),
    fullOutfit(Categories,Full).
    





    
