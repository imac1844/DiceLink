
Hooks.on('init', () => {
//     console.log(globalThis)
    game.settings.register("DiceLink", "diceLink", {
        name: "DiceLink",
        hint: "Connects to the DiceLink Python program",
        scope: "client",
        config: true,
        default: false,
        type: Boolean,
    });
});



Hooks.on('init', () => {
    function DiceLinkIntercept (evaluate, minimize=false, maximize=false, async) {
        let result = evaluate({minimize, maximize, async}={});
        return result
	};
    libWrapper.register(DiceLink, 'Game.Roll.evaluate', DiceLinkIntercept, 'WRAPPER');
});



// Hooks.on('roll', () => {
// 	console.log('Rolled a Die');
// 	console.log(this);
// });


// Hooks.on('baseRoll', () => {
// 	console.log('Rolled!')

// 	const baseRoll = Die.prototype.roll;
// 	// const prompt = require("prompt-async");
// 	Die.prototype.roll = async function(minimize=false, maximize=false) {
// 		if (game.settings.get('DiceLink','diceLink')) {
// 			const promptText = this.number > 1 ? 'Roll '+this.number+'d'+this.faces+' #'+(this.results.length + 1) : 'Roll a d'+this.faces;
// 			this.results.push({result: Number(prompt(promptText)), active: true});
// 			return this;
// 		}
// 		else {
// 			const roll = baseRoll.bind(this);
// 			roll(minimize, maximize);
// 		}

// });