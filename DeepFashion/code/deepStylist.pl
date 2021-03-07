% nn(neuralCategory,[I],C,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]) 
%     :: neuralCategory(I,C).
% nn(neuralPatternA,[I],P,[floral,graphic,striped,embroiderd,pleated,solid,lattice]) :: neuralPatternA(I,P).
% nn(neuralSleeveA,[I],S,[long_sleeve,short_sleeve,sleeveless]) :: neuralSleeveA(I,S).
% nn(neuralDLengthA,[I],L,[maxi_length,mini_length,no_dress]) :: neuralDLengthA(I,L).
% nn(neuralNeckA,[I],N,[crew_neckline,v_neckline,square_neckline,no_neckline]) :: neuralNeckA(I,N).
nn(neuralFabricA,[I],F,[denim,chiffon,cotton,leather,faux,knit]) :: neuralFabricA(I,F).
% nn(neuralFitA,[I],F,[tight,loose,conventional]) :: neuralFit(I,F).

% WEATHER ---------------------------------------------------------------------------------------------------------------

attributeTemperature(A,warm) :- member(A,[short_sleeve,sleeveless,denim,cotton,chiffon]).
attributeTemperature(A,cold) :- member(A,[long_sleeve,leather,faux,knit]).
% Apparently cotton is not good for cold https://www.quora.com/Is-cotton-good-for-winter
% Apparently denim is not good for cold https://www.theguardian.com/fashion/2020/sep/30/denim-is-rubbish-for-keeping-you-warm-experts-tips-for-cold-weather-dressing

attributePercipitation(A,rain) :- member(A,[denim,leather,faux]).
attributePercipitation(A,sun) :- member(A,[chiffon,cotton,knit,]).

attributeWeather(I,Temp,Percipitation) :- 
    neuralFabricA(I,F),
    attributeTemperature(F,Temp),
    attributePercipitation(F,Temp).
    % neuralSleeveA(I,S),
    % attributeTemperature(S,Temp).

% DRESSCODE ---------------------------------------------------------------------------------------------------------------

categoryDressCode(C,casual) :- member(C,[1,4,6,7,9,10,11,12,15,17,18,19,21,23,24,25,26,27,28,29,30,31,32,33,34,35,36,40,44,45,46,47,49,50]).
categoryDressCode(C,business) :- member(C,[2,3,5,11,14,22,39,47]).
categoryDressCode(C,chic) :- member(C,[2,5,6,8,13,16,20,23,33,37,38,41,42,45,47,48]).

% TODO: try learning probabilistic parameters for categories over multiple dresscodes (such as a jacket is most of the time (0.6) casual but sometimes business (0.4))
% t(0.6) :: business ... 

% Categories: casual, business, chic
%use categories (no tshirts for example) 
dresscode(I1,I2,Code) :- 
    neuralCategory(I1,C1), 
    neuralCategory(I2,C2), 
    categoryDressCode(C1,Code),
    categoryDressCode(C2,Code).

% For single pieces of clothing (full pieces like dresses or jumpsuits)
dressCode(I,Code) :- 
    neuralCategory(I,C),
    categoryDressCode(C,Code).

    
% SIMILAR ---------------------------------------------------------------------------------------------------------------

similar(Piece,SimilarPiece) :- 
    neuralCategory(Piece,C),
    % Get a few attributes here
    % TODO: Choose an images in database (that don't have labels, not needed or can update database as they are queried)
    %       I think a good idea here might be to choose a set of images to choose from and pick max from them (i.e. similar(Piece,SimilarPiece,SetOfPiecesToLookAt)). 
    %       In real-life application: these sets could be partitions of total dataset and these partitions can be run in parallel to get global, most similar piece.
    neuralCategory(SimilarPiece,C),
    neuralAttribute(SimilarPiece,A).
    % TODO: Potentially, show image here via IO.


% BUDGET ------------------------------------------------------------------------------------------------------------------

piecePrice(Piece,PiecePrice) :- True. % TODO, could be based on similarity to clothes of known price or could give price distributions of  

outfitPrice([Piece|PS],OutfitPrice) :- 
    outfitPrice(Ps,RestPrice), 
    OutfitPrice is RestPrice + piecePrice(Piece).

withinBudget(Pieces,Budget) :- outfitPrice(Pieces,OutfitPrice), Budget >= OutfitPrice.
