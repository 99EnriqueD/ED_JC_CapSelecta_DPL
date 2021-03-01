nn(neuralCategory,[I],C,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49]) 
    :: neuralCategory(I,C).
nn(neuralPatternA,[I],P,[floral,graphic,striped,embroiderd,pleated,solid,lattice]) :: neuralPatternA(I,P).
nn(neuralSleeveA,[I],S,[long_sleeve,short_sleeve,sleeveless]) :: neuralSleeveA(I,S).
nn(neuralLengthA,[I],L,[maxi_length,mini_length,no_dress]) :: neuralLengthA(I,L).
nn(neuralNeckA,[I],N,[crew_neckline,v_neckline,square_neckline,no_neckline]) :: neuralNeckA(I,N).
nn(neuralFabricA,[I],F,[denim,chiffon,cotton,leather,faux,knit]) :: neuralFabricA(I,F).
nn(neuralFitA,[I],F,[tight,loose,conventional]) :: neuralFit(I,F).

% WEATHER ---------------------------------------------------------------------------------------------------------------

attributeTemperature(A,warm) :- member(A,[short_sleeve,sleeveless,denim,cotton,chiffon]).
attributeTemperature(A,cold) :- member(A,[long_sleeve,leather,faux,cotton,denim,knit]).

attributePercipitation(A,rain) :- member(A,[denim,leather,faux]).
attributePercipitation(A,sun) :- member(A,[chiffon,cotton,knit,]).

attributeWeather(I,Temp,Percipitation) :- 
    neuralFabricA(I,F),
    attributeTemperature(F,Temp),
    attributePercipitation(F,Temp),
    
    neuralSleeveA(I,S),
    attributeTemperature(S,Temp).
    % attributePercipitation(S,Temp).

% DRESSCODE ---------------------------------------------------------------------------------------------------------------


% Categories: casual, business, chic
%use categories (no tshirts for example) 
dresscode(I1,I2,Code) :- 
    neuralCategory(I1,C1), 
    neuralCategory(I2,C2), 
    (
    % dress code for category of the two pieces are the same and are equal to the outfit's dresscode
    categoryDressCode(C1,Code),
    categoryDressCode(C2,Code)
    ;
    % dress code for category of the two pieces is mixed
    Code = mixed
    ).

% For single pieces of clothing (full pieces like dresses or jumpsuits)
dressCode(I,Code) :- 
    neuralCategory(I,C),
    categoryDressCode(C,Code).

    

categoryDressCode(C,casual) :- member(C,).
categoryDressCode(C,business) :- member(C,).
categoryDressCode(C,)

% SIMILAR ---------------------------------------------------------------------------------------------------------------

%use combination of 
similar(Piece,SimilarPiece) :- 
    neuralCategory(Piece,C),
    neuralAttribute(Piece,A),
    % TODO: Choose an images in database (that don't have labels, not needed or can update database as they are queried)
    neuralCategory(SimilarPiece,C),
    neuralAttribute(SimilarPiece,A).
    % TODO: Potentially, show image here via IO.