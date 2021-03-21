nn(fashion_df_net,[X], O, [0,1,2,3,4,5,6]) :: category(X,O).

memberchk(El, L) :- once(member(El,L)).
%%%%%%%
member(El, [H|T]) :- member_(T, El, H).
member_(_, El, El).
member_([H|T], El,_) :- member_(T, El, H).
%%%%%%%
%shirt achtig 0 
%trui achtig 1
%short 2
%broek casual 3
%broek normaal 4
%skirt + dress 5
%coat 6

% Wardrobe must include a coat or pulloverthingy.
goodForRain(Pieces,1) :- memberchk(6,Pieces).
goodForRain(Pieces,1) :- memberchk(1,Pieces).
goodForRain(Pieces,0) :- \+goodForRain(Pieces,1).

% Wardrobe must not include casual trousers or a pulloverthingy.
formal(Pieces,0) :- memberchk(3,Pieces).
formal(Pieces,0) :- memberchk(1,Pieces).
formal(Pieces,1) :- \+formal(Pieces,0).

% Whether the pieces int the given wardrobe make a full outfit.
fullOutfit(Pieces,1) :- hasTop(Pieces), hasBottoms(Pieces).
fullOutfit(Pieces,0) :- \+fullOutfit(Pieces,1).

%hasTop(Pieces) :-  member(X,Pieces), member(X,[0,1,6]).
%hasBottoms(Pieces) :- member(X,Pieces), member(X,[2,3,4,5]).

hasTop(Pieces) :- memberchk(0,Pieces). 
hasTop(Pieces) :- memberchk(1,Pieces).
hasTop(Pieces) :- memberchk(6,Pieces).
hasBottoms(Pieces) :- memberchk(2,Pieces).
hasBottoms(Pieces) :- memberchk(3,Pieces).
hasBottoms(Pieces) :- memberchk(4,Pieces).
hasBottoms(Pieces) :- memberchk(5,Pieces).


appropriateWardrobe(I1, I2, Rain, Formal, Full) :- 
    category(I1,C1),
    category(I2,C2),
    goodForRain([C1,C2],Rain),
    formal([C1,C2],Formal),
    fullOutfit([C1,C2],Full).
